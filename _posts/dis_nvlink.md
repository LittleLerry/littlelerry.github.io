



# Dissecting NVLink: Latency and the L2 Cache

**TL;DR**

1. A GPU's L2 can serve requests arriving from its own GPCs and from NVLink.
2. Even when the physical address is the same, the way L2 caches the data can differ. Both the issued instruction and the route the request takes matter.
3. Requests arriving through NVLink, and atomic requests, seem less likely to create local caching node (LCN) copies than loads coming from a local GPC. That would put more L2 capacity pressure on local loads. The details of how those copies are created remain a hypothesis.
4. If this picture is right, it might also create a side-channel opportunity: a remote observer could fill PoC entries, then look for changes in reload latency after a victim's local loads create LCN copies. We have not tested that possibility.

## 1. Measuring Latency

NVLink bandwidth tells us how much data the link can move. It does not tell us how long one thread waits before it can use the result of a remote load. So we start with an access pattern that deliberately avoids memory-level parallelism: one thread on one SM follows a linked list in GPU memory, reading each node in turn. Every node contains a 64-bit value holding the byte offset of the next node from the start of the allocation.

Here is the basic idea:

```text
offset = 0
checksum = 0
start = timer_on_this_GPU()

repeat N times:
    value = load_u64(base + offset)  # base + offset may point to local or peer GPU memory
    offset = low_32_bits(value)      # the next address depends on this returned value
    checksum += value

consume(checksum)                    # force the final returned value to be used
stop = timer_on_this_GPU()

check offset == 0
check checksum == expected_checksum
latency = (stop - start) / N
```

With the GPUs in the same LSA and peer access established, `load_u64` can address memory on another GPU. Point `base` at a local allocation to measure a local load, or at a preallocated region on the other GPU to measure a peer load.

What we call *latency* is the average cost of a full dependent-loop iteration. It includes waiting for the load result and doing the address and checksum work. It is not an isolated wire latency.

For the experiments below, we use one contiguous **64 MiB** allocation. We choose N nodes from positions spaced **128 B** apart. That gives us 64 MiB / 128 B = 524,288 candidate positions. We shuffle the chosen nodes into a ring, with each node pointing to the next. The 128 B spacing is meant to keep neighboring nodes from benefiting from the same cache line. A **full cycle** visits every selected node once and returns to the start. At 256K nodes (**K = 1024**), the loads return only 2 MiB of logical data in total (256K × 8 B), but those loads touch 256K separate 128 B address regions.

We call the GPU that owns the allocation **A**. Another GPU that can access it is **B**, the peer requester. When A reads its own allocation, that is a **local load, A→A**. When B reads A's allocation, that is a **peer load, B→A**.

The measurements in this article come from multiple fresh processes. Each process collects 30 timed samples, and we report averages of the process means.

## 2. Load Latency

### 2.1 Load Instructions

There are several ways to express a load. We care about three of them here: `ld.global.cg.u64`, `ld.relaxed.gpu.global.u64`, and `ld.relaxed.sys.global.u64`. On our `sm_90` machine, they compile to the following SASS in the matched kernel. The PTX forms are defined in the [PTX load instruction specification](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld).

| PTX | Emitted SASS load | Difference from the `.cg` kernel |
|---|---|---|
| `ld.global.cg.u64` | `LDG.E.64.STRONG.GPU` | Baseline |
| `ld.relaxed.gpu.global.u64` | `LDG.E.64.STRONG.GPU` | All 64 instruction encodings are identical |
| `ld.relaxed.sys.global.u64` | `LDG.E.64.STRONG.SYS` | Only this load instruction differs |

In this kernel, `ld.relaxed.gpu.global.u64` and `ld.global.cg.u64` really do emit the same SASS. The SYS version changes the load's scope field. No extra fence instruction appears in the SYS kernel.

### 2.2 How a Peer Load Is Served

When B executes `ld.global.cg.u64` against A's allocation, it bypasses B's L1. The request goes over NVLink and is served by A's L2 or, if needed, A's HBM. This sketch shows the memory levels we care about:

```text
B issues a .cg load
    → request travels over NVLink
    → hit in A's L2: data returns to B
    → miss in A's L2: fetch from A's HBM, then return to B
```

NVIDIA's [GH100 discussion](https://forums.developer.nvidia.com/t/ncu-l2-metric-understanding/361731) describes how requests arriving over NVLink reach the owner-side memory system. We can also test the importance of A's state on our machine. We let B follow a 4,096-node ring in A's memory. After evicting the target data, B's first full traversal averages **968.80 ns per step**. If A first reads that ring with `ld.global.cg.u64`, B's traversal drops to **813.42 ns per step**. If B warms the ring itself, its following traversal takes **813.57 ns per step**. Warming on A helps B even though B did not first traverse the peer ring.

With `.ca`, the picture can be different. A small ring may be served from B's L1 on repeated accesses. When that does not happen, B requests the data over NVLink; a response from A's L2 or HBM can then populate B's L1. That gives us a straightforward test: grow the ring until requester-side reuse stops helping.

```text
B issues a .ca load
    → hit in B's L1: return the value directly
    → otherwise send a request to A over NVLink
    → hit in A's L2: return the value and cache it in B's L1
    → otherwise fetch from A's HBM, return it, and cache it in B's L1
```

| Peer pair | Ring nodes | `.cg` ns/step | `.ca` ns/step | `.nc` ns/step |
|---|---:|---:|---:|---:|
| g0, B=2 → A=3 | 1 | 844.53 | 31.87 | 31.87 |
| Same pair | 32 | 811.39 | 33.34 | 33.34 |
| Same pair | 4,096 | 813.72 | 812.49 | 812.49 |
| Second window, B=0 → A=1 | 1 | 885.27 | 31.87 | 31.87 |
| Same pair | 4,096 | 866.57 | 864.70 | 864.70 |

![Requester-side reuse by load instruction](/assets/_posts/dis_nvlink/img/r01-cache_operators.png)

At one or 32 nodes, `.ca` and `.nc` are much faster than `.cg`. By 4,096 scattered nodes, that advantage is mostly gone. Absolute `.cg` latency also differs between GPU pairs, but the within-pair comparison is what matters here. The result means a peer load can reuse a requester-side cached value without a new NVLink request. If the owner later changes that value, B could read an older local copy instead of seeing the update immediately.

### 2.3 Local and Peer Loads Affect L2 Caching Differently

#### 2.3.1 The Local and Peer Latency Curves Split

We first saw the difference in an early latency sweep:

![Local and peer load latency for the same owner](/assets/_posts/dis_nvlink/img/r01-same_owner_load_w1.png)

The left plot shows local `ld.global.cg.u64` latency as we increase the number of nodes. At 128K nodes, it stays below 200 ns. As the node count rises, so does the latency: more accesses seem to outgrow the fast cache-serving regime and require more expensive service. The right plot repeats the idea with a peer GPU issuing the loads. Its latency eventually rises too.

The striking part is **where those rises happen**. At 128K nodes, we measure about **168 ns locally** and **814 ns from the peer**. At 256K, the local load has climbed to about **280 ns**, while the peer load is still around **815 ns**. At 512K, both are more expensive: about **368 ns locally** and **969 ns from the peer**.

The access pattern is the same. The local load reaches the expensive regime earlier, while the peer load does not. A fixed NVLink round-trip added to the local curve cannot explain that shape. It suggests that A's L2 responds differently to requests arriving locally and requests arriving over NVLink.

We checked whether the load's scope could be responsible. Local and peer loads were each tested with `ld.relaxed.gpu.global.u64` and `ld.relaxed.sys.global.u64`, giving four scope pairings:

![Local and peer latency under four load-scope pairings](/assets/_posts/dis_nvlink/img/r05-scope_panels_first.png)

The same separation appears in every pairing. For the same issuing GPU and node count, the largest measured first-cycle difference between SYS and GPU scope is only **0.29 ns**. The local-versus-peer difference follows where the request originates, not a switch between these two scope encodings.

#### 2.3.2 A Capacity-Pressure Model: PoC and LCN

There are several ways the cache could produce this behavior. We briefly considered whether NVLink has its own L2 buffer or a more complicated path through L2. A more useful lead came from the **Point of Coherence (PoC)** and **Local Caching Node (LCN)** model.

Think of PoC as the L2 location responsible for an address `x`. The LCN can keep a copy closer to a requesting SM. If the PoC for `x` sits in a distant L2 partition, the GPU may choose to keep another copy nearer the SM. The copy near the SM is the LCN copy; the PoC remains responsible for coherence. The hardware has to keep the two consistent.

NVIDIA explains that a value can live in both the PoC and an LCN, consuming extra effective L2 capacity ([PoC/LCN discussion](https://forums.developer.nvidia.com/t/use-of-l2-cache/328183/10)). A cross-partition request can reach a local L2 slice first and then cross the L2 fabric to the PoC slice ([H100 architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/), [GH100 discussion](https://forums.developer.nvidia.com/t/ncu-l2-metric-understanding/361731)). We suspect these roles may be represented in line metadata, perhaps somewhat like tags or valid bits, but that part is only a guess. The third-party [Citadel GTC 2021 talk](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s33322/?_gl=1*p9km09*_gcl_au*MjU0MzUwNTQuMTc4NTA0OTI0Ny4tLi0uMTc4ODkxMjk5MC44MjI2ODQ1ODkuMTc4OTAxNTAxMC4xNzg5MDE1MTM5) uses A100 near and far L2 access to illustrate a similar tradeoff.

Now apply that idea to our measurements. At 256K nodes with 128 B spacing, the active addresses cover a 32 MiB span. That looks smaller than the H800's 50 MiB L2. But if one SM needs LCN copies for many addresses whose PoCs sit farther away, those copies add pressure to a local partition. That partition can run short of space before the aggregate L2 does.

This might explain why local loads hit the capacity-related slowdown earlier than peer loads: A's L2 may be less inclined to make the same near-SM copies for a request arriving over NVLink. The peer path could therefore *appear* to have more usable L2 capacity. It is also a reasonable performance tradeoff: a nearby copy saves a larger fraction of a local SM load's latency than it saves for a load that already pays a substantial remote-access cost. That tradeoff is our explanation, not a measured hardware policy.

The next experiments look for more evidence.

#### 2.3.3 Local L2-Fabric and HBM Traffic

We first ask what happens to L2-fabric activity and HBM traffic as we increase the node count. These counters are collected on A, the owner GPU.

**Local: A warms, A reads.**

| Nodes | First ns/step | HBM read MB | HBM write MB | Fabric hit sectors | Fabric miss sectors |
|---:|---:|---:|---:|---:|---:|
| 128K | 167.41 | 0.780 | 0.689 | 18,900.0 | 2.0 |
| 256K | 279.83 | 9.577 | 4.710 | 233,317.2 | 11,227.2 |
| 512K | 367.50 | 35.499 | 10.522 | 176.0 | 524,288.0 |

At 128K, HBM reads are low and this fabric counter reports almost no misses, consistent with most requests finding useful cached data. At 256K, HBM reads have risen, but so have fabric hits. The cache is under more pressure, and some data is going to HBM, while many requests still find data along the fabric path. At 512K, HBM reads rise again and this fabric counter reports about 524K miss sectors. The combination at 256K is the interesting part: more HBM reads **and** many fabric hits. It fits a picture where local cache advantages erode and more requests travel across partitions.

**Peer: B warms, B reads; the counters are still collected on A.**

The selected fabric counters look odd for peer reads:

| Nodes | First ns/step | HBM read MB | HBM write MB | Fabric hit sectors | Fabric miss sectors |
|---:|---:|---:|---:|---:|---:|
| 128K | 813.21 | 0.426 | 1.994 | 80.0 | 0.0 |
| 256K | 814.47 | 1.586 | 4.502 | 80.0 | 0.0 |
| 512K | 969.00 | 20.584 | 16.399 | 80.0 | 0.0 |

Peer latency stays around 814 ns at 128K and 256K, and owner HBM reads are far below the local-load case at the same node count. At 512K, both latency and owner HBM reads rise sharply. This suggests that the peer can use A's L2 effectively at intermediate sizes, but relies more on HBM at the largest size.

The fabric hit/miss counts, however, stay close to their background values. A control makes that limitation clear: a cold 128K peer traversal produces about **5.10 MB** of owner HBM reads, compared with about **0.42 MB** after peer warming. Both show **80 fabric-hit sectors and zero fabric-miss sectors** in this metric class. So the HBM counter detects a real difference, while these selected fabric counters do not show the corresponding peer activity.

#### 2.3.4 Changing Who Warms the Data

To test our idea further, we change who warms the data, then measure the next load. At 128K nodes, our model predicts little overall capacity pressure from PoC plus LCN copies:

1. If A warms the ring, the copies it creates should still fit. B's following load should be about as fast as it would be after B warmed the ring itself.
2. If B warms the ring, we expect PoC copies but few, perhaps no, LCN copies near A's SM. A may have to create those copies during its first traversal. Its first cycle should be slower, while its second should approach the latency after A-warming. The first cycle should still beat a completely cold HBM traversal, because B's warming has done some useful work.

The 128K results line up with these predictions. Each cell shows **first → second** traversal latency in ns/step:

**128K nodes**

| Warmup ↓ / Probe → | A `.cg` (local) | B `.cg` (peer) |
|---|---:|---:|
| A `.cg` | 167.41 → 167.28 | 813.49 → 813.00 |
| B `.cg` | 286.64 → 167.45 | 813.21 → 813.21 |

At 256K nodes, the model says PoC plus LCN copies have started to press on L2, while PoC copies alone are not yet under the same pressure:

1. After A warms the ring, B's first cycle should be slower as it sheds state that helped A. Once B has traversed the ring, its second cycle should drop toward B's usual warm latency.
2. After B warms the ring, A has to build useful local copies while also dealing with eviction. Its first cycle should be slow, though still faster than a cold HBM traversal. Its second cycle should return toward A's usual warm latency.

Again, that is what the 256K results look like:

**256K nodes**

| Warmup ↓ / Probe → | A `.cg` (local) | B `.cg` (peer) |
|---|---:|---:|
| A `.cg` | 279.83 → 279.67 | 849.37 → 814.57 |
| B `.cg` | 322.81 → 279.75 | 814.47 → 814.41 |

At 512K, our model expects more pressure on the PoC copies themselves. In this simple picture, most entries get pushed out quickly, so changing the warming GPU does little for latency:

**512K nodes**

| Warmup ↓ / Probe → | A `.cg` (local) | B `.cg` (peer) |
|---|---:|---:|
| A `.cg` | 367.50 → 367.50 | 968.90 → 968.90 |
| B `.cg` | 367.84 → 367.83 | 969.00 → 969.00 |

## 3. TMA Load Latency

We see a similar origin-dependent pattern with TMA loads. In the configurations we measured, how a TMA request affects later L2 behavior seems to depend mainly on whether that request came over NVLink.

For a local TMA copy, a `cp.async.bulk.tensor.1d` operation goes to the TMA hardware and moves data from local global memory into local shared memory. For a peer TMA copy, the tensor map instead describes an address in the peer-accessible region of A's memory. The hardware routes the request over NVLink and handles the data movement. It makes the interface pleasantly simple, though the program still needs to wait for completion correctly.

```text
B issues a peer TMA load
    → the request goes over NVLink to A
    → hit in A's L2: data returns to B's TMA path
    → miss: fetch from A's HBM, then return to B
    → B's TMA writes B's shared memory and completes the mbarrier protocol
```

We tested how different warmup methods affect later probes of A's data. The TMA path has no explicit cache hint in this experiment; it uses rank 1 to load 16 B of 1D data per selected node. The SM16 control warms the same 16 B with `ld.global.cg.v2.u64`.

Here are the first-probe results:

![First probe traversal after TMA or matched SM preparation](/assets/_posts/dis_nvlink/img/tma-first_cycle.png)

Each panel shows a different probe. A.TMA means TMA on A warmed A's L2; B.TMA means TMA on B accessed A's allocation over NVLink. A.SM16 and B.SM16 use the matching 16 B SM load instead. Blue marks preparation originating on A, orange preparation originating on B. With the probe held fixed, the curves mostly group by **which GPU performed the warmup**. The TMA and SM16 versions of the same origin nearly overlap; their measured differences are on the order of one SM clock cycle:

![TMA versus SM preparation: first-probe latency difference](/assets/_posts/dis_nvlink/img/tma-first_preparation_difference.png)

The second-probe results overlap even more closely:

![Second probe traversal after the same preparation](/assets/_posts/dis_nvlink/img/tma-second_cycle.png)

For this workload, TMA warming leaves an L2 state very much like warming with the matched ordinary SM load. Our capacity-pressure model can also describe that behavior.

## 4. Atomic Latency

Atomics in this experiment look more like peer loads: they seem less inclined to create LCN copies. We ran another warmup/probe matrix, and the atomic results resemble the peer-load results:

![Preparation changes the cost of the same probe](/assets/_posts/dis_nvlink/img/r03-requested_matrix.png)

That gives us a plausible reason to suspect fewer LCN copies on the atomic path. If hardware already has to keep PoC and LCN copies coherent, maintaining more local copies for an atomic read-modify-write might be too expensive. That is still a hypothesis about the mechanism, rather than a direct count of LCN copies.

## 5. Conclusions and Implications

The strongest picture we can draw from these experiments is that L2 behavior depends on whether a request arrives from NVLink or a local GPC, as well as on the operation being performed. For NVLink load requests, L2 may be less inclined to create LCN copies; for loads from a local GPC, it may prefer to keep those local copies. The resulting cache state changes the latency of later operations, and our probes can see that change.

That could also open the door to a side channel. Imagine a remote GPU filling as many PoC entries as it can with loads. If a victim's local loads create LCN copies that displace some of those PoC entries, the remote GPU might detect the evictions through slower reloads. In principle, that could reveal something about the victim's physical memory access pattern. This is a possible implication of the model, not a channel we have demonstrated.
