#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#include <vulkan/vulkan.h>

#define CHECK_VK(call) do { \
    VkResult _r = (call); \
    if (_r != VK_SUCCESS) { \
        fprintf(stderr, "FAIL: %s returned %d\n", #call, (int)_r); \
        exit(1); \
    } \
} while (0)

typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void *mapped;
} HostBuffer;

static uint32_t find_memory_type(
    VkPhysicalDevice physical,
    uint32_t type_bits,
    VkMemoryPropertyFlags flags
) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(physical, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) != 0 && (props.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    fprintf(stderr, "FAIL: no suitable memory type\n");
    exit(1);
}

static HostBuffer make_host_buffer(
    VkPhysicalDevice physical,
    VkDevice device,
    VkDeviceSize bytes
) {
    HostBuffer out;
    memset(&out, 0, sizeof(out));

    VkBufferCreateInfo buffer_info;
    memset(&buffer_info, 0, sizeof(buffer_info));
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = bytes;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHECK_VK(vkCreateBuffer(device, &buffer_info, NULL, &out.buffer));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, out.buffer, &req);

    VkMemoryAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = req.size;
    alloc_info.memoryTypeIndex = find_memory_type(
        physical,
        req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    CHECK_VK(vkAllocateMemory(device, &alloc_info, NULL, &out.memory));
    CHECK_VK(vkBindBufferMemory(device, out.buffer, out.memory, 0));
    CHECK_VK(vkMapMemory(device, out.memory, 0, bytes, 0, &out.mapped));
    return out;
}

static void destroy_host_buffer(VkDevice device, HostBuffer *buffer) {
    if (buffer->mapped) {
        vkUnmapMemory(device, buffer->memory);
    }
    if (buffer->buffer) {
        vkDestroyBuffer(device, buffer->buffer, NULL);
    }
    if (buffer->memory) {
        vkFreeMemory(device, buffer->memory, NULL);
    }
}

static int looks_like_binary_spv(const unsigned char *bytes, size_t n) {
    return n >= 4 && bytes[0] == 0x03 && bytes[1] == 0x02 && bytes[2] == 0x23 && bytes[3] == 0x07;
}

static uint32_t *parse_decimal_spv_words(const char *path, const char *text, size_t n, size_t *bytes_out) {
    size_t count = 0;
    const char *p = text;
    const char *end = text + n;
    while (p < end) {
        while (p < end && isspace((unsigned char)*p)) p++;
        if (p >= end) break;
        char *next = NULL;
        strtoul(p, &next, 10);
        if (next == p) {
            fprintf(stderr, "FAIL: invalid decimal SPIR-V word in %s\n", path);
            exit(1);
        }
        count++;
        p = next;
    }
    if (count == 0) {
        fprintf(stderr, "FAIL: no decimal SPIR-V words in %s\n", path);
        exit(1);
    }

    uint32_t *words = (uint32_t *)malloc(count * sizeof(uint32_t));
    if (!words) {
        fprintf(stderr, "FAIL: malloc decimal SPIR-V\n");
        exit(1);
    }
    p = text;
    size_t i = 0;
    while (p < end && i < count) {
        while (p < end && isspace((unsigned char)*p)) p++;
        if (p >= end) break;
        char *next = NULL;
        unsigned long word = strtoul(p, &next, 10);
        words[i++] = (uint32_t)word;
        p = next;
    }
    if (words[0] != 0x07230203u) {
        fprintf(stderr, "FAIL: decimal SPIR-V magic word missing in %s\n", path);
        exit(1);
    }
    *bytes_out = count * sizeof(uint32_t);
    return words;
}

static uint32_t *read_spv(const char *path, size_t *bytes_out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "FAIL: cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    rewind(f);
    if (n <= 0) {
        fprintf(stderr, "FAIL: invalid SPIR-V byte size %ld\n", n);
        exit(1);
    }
    unsigned char *raw = (unsigned char *)malloc((size_t)n + 1);
    if (!raw) {
        fprintf(stderr, "FAIL: malloc SPIR-V input\n");
        exit(1);
    }
    if (fread(raw, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "FAIL: fread SPIR-V\n");
        exit(1);
    }
    fclose(f);
    raw[n] = 0;

    if (looks_like_binary_spv(raw, (size_t)n)) {
        if ((n % 4) != 0) {
            fprintf(stderr, "FAIL: invalid binary SPIR-V byte size %ld\n", n);
            exit(1);
        }
        *bytes_out = (size_t)n;
        return (uint32_t *)raw;
    }

    uint32_t *words = parse_decimal_spv_words(path, (const char *)raw, (size_t)n, bytes_out);
    free(raw);
    return words;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <storage_vec_add.spv>\n", argv[0]);
        return 2;
    }

    const uint32_t n = 64;
    const VkDeviceSize bytes = n * sizeof(float);

    VkApplicationInfo app;
    memset(&app, 0, sizeof(app));
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "kretikos_spirv_vulkan_storage_vec_add";
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instance_info;
    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app;

    VkInstance instance = VK_NULL_HANDLE;
    CHECK_VK(vkCreateInstance(&instance_info, NULL, &instance));

    uint32_t physical_count = 0;
    CHECK_VK(vkEnumeratePhysicalDevices(instance, &physical_count, NULL));
    if (physical_count == 0) {
        fprintf(stderr, "FAIL: no Vulkan device\n");
        return 1;
    }
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    CHECK_VK(vkEnumeratePhysicalDevices(instance, &physical_count, &physical));

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical, &props);

    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, NULL);
    VkQueueFamilyProperties *families = calloc(family_count, sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families);
    uint32_t compute_family = UINT32_MAX;
    for (uint32_t i = 0; i < family_count; i++) {
        if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
            compute_family = i;
            break;
        }
    }
    free(families);
    if (compute_family == UINT32_MAX) {
        fprintf(stderr, "FAIL: no compute queue\n");
        return 1;
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info;
    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = compute_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;

    VkDeviceCreateInfo device_info;
    memset(&device_info, 0, sizeof(device_info));
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;

    VkDevice device = VK_NULL_HANDLE;
    CHECK_VK(vkCreateDevice(physical, &device_info, NULL, &device));

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, compute_family, 0, &queue);

    HostBuffer a = make_host_buffer(physical, device, bytes);
    HostBuffer b = make_host_buffer(physical, device, bytes);
    HostBuffer c = make_host_buffer(physical, device, bytes);

    float *ap = (float *)a.mapped;
    float *bp = (float *)b.mapped;
    float *cp = (float *)c.mapped;
    for (uint32_t i = 0; i < n; i++) {
        ap[i] = (float)((int)(i % 17) - 8) * 0.25f;
        bp[i] = (float)((int)(i % 11) - 5) * 0.5f;
        cp[i] = 0.0f;
    }

    VkDescriptorSetLayoutBinding bindings[3];
    memset(bindings, 0, sizeof(bindings));
    for (uint32_t i = 0; i < 3; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dsl_info;
    memset(&dsl_info, 0, sizeof(dsl_info));
    dsl_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_info.bindingCount = 3;
    dsl_info.pBindings = bindings;

    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    CHECK_VK(vkCreateDescriptorSetLayout(device, &dsl_info, NULL, &dsl));

    VkPipelineLayoutCreateInfo layout_info;
    memset(&layout_info, 0, sizeof(layout_info));
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &dsl;

    VkPipelineLayout layout = VK_NULL_HANDLE;
    CHECK_VK(vkCreatePipelineLayout(device, &layout_info, NULL, &layout));

    size_t spv_bytes = 0;
    uint32_t *spv = read_spv(argv[1], &spv_bytes);
    VkShaderModuleCreateInfo module_info;
    memset(&module_info, 0, sizeof(module_info));
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = spv_bytes;
    module_info.pCode = spv;

    VkShaderModule shader = VK_NULL_HANDLE;
    CHECK_VK(vkCreateShaderModule(device, &module_info, NULL, &shader));

    VkPipelineShaderStageCreateInfo stage;
    memset(&stage, 0, sizeof(stage));
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName = "main";

    VkComputePipelineCreateInfo pipeline_info;
    memset(&pipeline_info, 0, sizeof(pipeline_info));
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage;
    pipeline_info.layout = layout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    CHECK_VK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline));

    VkDescriptorPoolSize pool_size;
    memset(&pool_size, 0, sizeof(pool_size));
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 3;

    VkDescriptorPoolCreateInfo pool_info;
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;

    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    CHECK_VK(vkCreateDescriptorPool(device, &pool_info, NULL, &descriptor_pool));

    VkDescriptorSetAllocateInfo set_alloc;
    memset(&set_alloc, 0, sizeof(set_alloc));
    set_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_alloc.descriptorPool = descriptor_pool;
    set_alloc.descriptorSetCount = 1;
    set_alloc.pSetLayouts = &dsl;

    VkDescriptorSet set = VK_NULL_HANDLE;
    CHECK_VK(vkAllocateDescriptorSets(device, &set_alloc, &set));

    VkDescriptorBufferInfo infos[3];
    memset(infos, 0, sizeof(infos));
    infos[0].buffer = a.buffer; infos[0].range = bytes;
    infos[1].buffer = b.buffer; infos[1].range = bytes;
    infos[2].buffer = c.buffer; infos[2].range = bytes;

    VkWriteDescriptorSet writes[3];
    memset(writes, 0, sizeof(writes));
    for (uint32_t i = 0; i < 3; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device, 3, writes, 0, NULL);

    VkCommandPoolCreateInfo cmd_pool_info;
    memset(&cmd_pool_info, 0, sizeof(cmd_pool_info));
    cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmd_pool_info.queueFamilyIndex = compute_family;

    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    CHECK_VK(vkCreateCommandPool(device, &cmd_pool_info, NULL, &cmd_pool));

    VkCommandBufferAllocateInfo cmd_alloc;
    memset(&cmd_alloc, 0, sizeof(cmd_alloc));
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = cmd_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    CHECK_VK(vkAllocateCommandBuffers(device, &cmd_alloc, &cmd));

    VkCommandBufferBeginInfo begin;
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VK(vkBeginCommandBuffer(cmd, &begin));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
    vkCmdDispatch(cmd, n, 1, 1);
    CHECK_VK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submit;
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    CHECK_VK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
    CHECK_VK(vkQueueWaitIdle(queue));

    float max_abs_err = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float expected = ap[i] + bp[i];
        float err = fabsf(cp[i] - expected);
        if (err > max_abs_err) max_abs_err = err;
    }
    if (max_abs_err > 0.000001f) {
        fprintf(stderr, "FAIL: max_abs_err=%g\n", max_abs_err);
        return 1;
    }

    printf(
        "kretikos_spirv_vulkan_storage_vec_add status=pass n=%u max_abs_err=%g device_name=%s api_version=%u.%u.%u\n",
        n,
        max_abs_err,
        props.deviceName,
        VK_VERSION_MAJOR(props.apiVersion),
        VK_VERSION_MINOR(props.apiVersion),
        VK_VERSION_PATCH(props.apiVersion)
    );

    CHECK_VK(vkDeviceWaitIdle(device));
    vkDestroyCommandPool(device, cmd_pool, NULL);
    vkDestroyDescriptorPool(device, descriptor_pool, NULL);
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyShaderModule(device, shader, NULL);
    free(spv);
    vkDestroyPipelineLayout(device, layout, NULL);
    vkDestroyDescriptorSetLayout(device, dsl, NULL);
    destroy_host_buffer(device, &c);
    destroy_host_buffer(device, &b);
    destroy_host_buffer(device, &a);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    return 0;
}
