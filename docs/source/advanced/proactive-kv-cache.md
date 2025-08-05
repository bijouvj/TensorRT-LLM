# Proactive KV Cache Manager

The Proactive KV Cache Manager is an extension to TensorRT-LLM's KV cache management system that proactively evicts KV blocks from HBM (High Bandwidth Memory) to make them available for prefill or filling from host DRAM immediately. This helps improve memory utilization and reduce latency during inference.

## Overview

The proactive KV cache manager monitors the availability of free blocks in both primary (HBM) and secondary (DRAM) memory pools and automatically performs eviction and preloading operations to maintain optimal memory utilization.

### Key Features

- **Proactive Eviction**: Automatically moves blocks from HBM to DRAM when free block thresholds are reached
- **Proactive Preloading**: Moves high-priority blocks from DRAM back to HBM when space becomes available
- **Configurable Thresholds**: Adjustable thresholds for triggering eviction and preloading operations
- **Priority-Based Management**: Uses block priorities to determine which blocks to evict or preload
- **Background Worker**: Runs eviction/preloading operations in a background thread to avoid blocking inference
- **Statistics Tracking**: Provides detailed statistics about eviction and preloading operations

## Architecture

The proactive KV cache manager wraps the standard `KVCacheManager` and adds proactive memory management capabilities:

```
ProactiveKVCacheManager
├── BaseKVCacheManager (interface)
├── KVCacheManager (base implementation)
├── Proactive Eviction Worker Thread
├── Configuration Management
└── Statistics Tracking
```

## Configuration

The proactive behavior is controlled through the `ProactiveConfig` structure:

```cpp
struct ProactiveConfig
{
    // Threshold for free blocks in primary memory before proactive eviction starts
    SizeType32 primaryFreeBlockThreshold = 10;
    
    // Threshold for free blocks in secondary memory before proactive eviction starts
    SizeType32 secondaryFreeBlockThreshold = 5;
    
    // Number of blocks to proactively evict when thresholds are met
    SizeType32 proactiveEvictionBatchSize = 5;
    
    // Minimum time between proactive eviction cycles (milliseconds)
    std::chrono::milliseconds minEvictionInterval = std::chrono::milliseconds(100);
    
    // Maximum time to wait for blocks to become available (milliseconds)
    std::chrono::milliseconds maxWaitTime = std::chrono::milliseconds(1000);
    
    // Whether to enable proactive eviction
    bool enableProactiveEviction = true;
    
    // Priority threshold for proactive eviction (blocks with lower priority will be evicted first)
    executor::RetentionPriority evictionPriorityThreshold = 50;
    
    // Whether to preload blocks from secondary to primary when space becomes available
    bool enablePreloading = true;
    
    // Number of blocks to preload when space becomes available
    SizeType32 preloadBatchSize = 3;
};
```

## Usage

### Basic Usage

```cpp
#include "tensorrt_llm/batch_manager/proactiveKvCacheManager.h"

// Configuration parameters
std::vector<SizeType32> numKvHeadsPerLayer = {32, 32, 32, 32, 32, 32};
SizeType32 sizePerHead = 128;
SizeType32 tokensPerBlock = 64;
SizeType32 blocksInPrimaryPool = 1000;
SizeType32 blocksInSecondaryPool = 2000;
SizeType32 maxNumSequences = 100;
SizeType32 maxBeamWidth = 1;
std::vector<SizeType32> maxAttentionWindowVec = {2048};

// Create CUDA stream
auto stream = std::make_shared<CudaStream>();

// Proactive configuration
ProactiveKVCacheManager::ProactiveConfig proactiveConfig;
proactiveConfig.primaryFreeBlockThreshold = 50;
proactiveConfig.secondaryFreeBlockThreshold = 100;
proactiveConfig.proactiveEvictionBatchSize = 10;
proactiveConfig.minEvictionInterval = std::chrono::milliseconds(200);
proactiveConfig.enableProactiveEviction = true;
proactiveConfig.evictionPriorityThreshold = 50;
proactiveConfig.enablePreloading = true;
proactiveConfig.preloadBatchSize = 5;

// Create proactive KV cache manager
auto proactiveKvCacheManager = std::make_unique<ProactiveKVCacheManager>(
    numKvHeadsPerLayer, sizePerHead, tokensPerBlock, blocksInPrimaryPool, blocksInSecondaryPool,
    maxNumSequences, maxBeamWidth, maxAttentionWindowVec, temporaryAttentionWindow, sinkTokenLength,
    stream, std::nullopt, true, true, CacheType::kSELF, std::nullopt, nullptr, false, true, true,
    proactiveConfig);

// Allocate pools
proactiveKvCacheManager->allocatePools(nvinfer1::DataType::kHALF, false);

// Use the manager like a regular KV cache manager
proactiveKvCacheManager->addSequence(requestId, inputLength, beamWidth);
proactiveKvCacheManager->addToken(requestId);
proactiveKvCacheManager->removeSequence(requestId);
```

### Advanced Usage

#### Dynamic Configuration

You can change the proactive configuration at runtime:

```cpp
// Update configuration
ProactiveKVCacheManager::ProactiveConfig newConfig;
newConfig.primaryFreeBlockThreshold = 30;
newConfig.proactiveEvictionBatchSize = 15;
proactiveKvCacheManager->setProactiveConfig(newConfig);
```

#### Manual Triggering

You can manually trigger proactive eviction cycles:

```cpp
// Force a proactive eviction cycle
proactiveKvCacheManager->triggerProactiveEviction();
```

#### Statistics Monitoring

Monitor the proactive behavior through statistics:

```cpp
auto stats = proactiveKvCacheManager->getProactiveStats();
std::cout << "Total proactive evictions: " << stats.totalProactiveEvictions << std::endl;
std::cout << "Total proactive preloads: " << stats.totalProactivePreloads << std::endl;
std::cout << "Total eviction time: " << stats.totalEvictionTime.count() << " ms" << std::endl;
std::cout << "Total preload time: " << stats.totalPreloadTime.count() << " ms" << std::endl;
std::cout << "Cycles triggered: " << stats.cyclesTriggered << std::endl;
```

## Eviction Strategy

The proactive eviction strategy works as follows:

1. **Threshold Monitoring**: The background worker continuously monitors the number of free blocks in both primary and secondary memory pools.

2. **Trigger Conditions**: Eviction is triggered when:
   - Free blocks in primary memory ≤ `primaryFreeBlockThreshold`
   - Free blocks in secondary memory ≤ `secondaryFreeBlockThreshold`

3. **Block Selection**: Blocks are selected for eviction based on:
   - **Priority**: Blocks with lower priority are evicted first
   - **Location**: Only blocks in primary memory are considered for eviction
   - **Batch Size**: Up to `proactiveEvictionBatchSize` blocks are evicted per cycle

4. **Rate Limiting**: Eviction cycles are limited by `minEvictionInterval` to prevent excessive overhead.

## Preloading Strategy

The proactive preloading strategy works as follows:

1. **Space Availability**: Preloading is triggered when there are sufficient free blocks in primary memory (≥ 2 × `primaryFreeBlockThreshold`).

2. **Block Selection**: Blocks are selected for preloading based on:
   - **Priority**: Blocks with higher priority are preloaded first
   - **Location**: Only blocks in secondary memory are considered for preloading
   - **Batch Size**: Up to `preloadBatchSize` blocks are preloaded per cycle

3. **Memory Efficiency**: Preloading only occurs when there is sufficient space to avoid unnecessary transfers.

## Performance Considerations

### Benefits

- **Reduced Latency**: Blocks are proactively moved to optimal memory locations
- **Better Memory Utilization**: More efficient use of both HBM and DRAM
- **Predictable Performance**: Reduces the likelihood of memory-related performance spikes
- **Configurable Overhead**: Adjustable thresholds and intervals allow tuning for specific workloads

### Trade-offs

- **Background Overhead**: The worker thread consumes CPU cycles
- **Transfer Overhead**: Moving blocks between memory pools has a cost
- **Configuration Complexity**: Requires tuning of multiple parameters for optimal performance

### Tuning Guidelines

1. **Thresholds**: Set thresholds based on your workload's memory usage patterns
   - Lower thresholds = more aggressive eviction = more overhead
   - Higher thresholds = less aggressive eviction = potential memory pressure

2. **Batch Sizes**: Balance between efficiency and responsiveness
   - Larger batches = more efficient transfers but less responsive
   - Smaller batches = more responsive but potentially more overhead

3. **Intervals**: Consider the frequency of your inference requests
   - Shorter intervals = more responsive but more overhead
   - Longer intervals = less overhead but potentially less responsive

4. **Priority Thresholds**: Align with your application's priority scheme
   - Lower threshold = more aggressive eviction of low-priority blocks
   - Higher threshold = more conservative eviction

## Integration with Existing Systems

The proactive KV cache manager is designed as a drop-in replacement for the standard `KVCacheManager`. It implements the same interface, so existing code can be easily modified to use proactive management:

```cpp
// Before: Using standard KV cache manager
auto kvCacheManager = std::make_unique<KVCacheManager>(...);

// After: Using proactive KV cache manager
auto kvCacheManager = std::make_unique<ProactiveKVCacheManager>(..., proactiveConfig);
```

## Example

See `examples/proactive_kv_cache_example.cpp` for a complete example demonstrating the usage of the proactive KV cache manager.

## Future Enhancements

Potential future enhancements include:

- **Adaptive Thresholds**: Automatically adjust thresholds based on workload characteristics
- **Predictive Preloading**: Use machine learning to predict which blocks will be needed
- **Multi-Level Memory**: Support for more than two memory levels
- **Workload-Specific Policies**: Different eviction strategies for different types of workloads
- **Integration with Schedulers**: Coordinate with request schedulers for better resource management 