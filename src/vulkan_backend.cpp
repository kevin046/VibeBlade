// VibeBlade Vulkan Backend — Cross-platform GPU compute via Vulkan
// Requires: Vulkan SDK 1.2+, SPIR-V shaders
// Build: CMake + Vulkan + pybind11

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <cstring>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// ─────────────────────────────────────────────
// Vulkan helper macros and RAII wrappers
// ─────────────────────────────────────────────

#define VK_CHECK(call) do { \
    VkResult _r = call; \
    if (_r != VK_SUCCESS) \
        throw std::runtime_error("Vulkan error " + std::to_string(_r) + \
                                 " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
} while(0)

struct VulkanBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
};

// ─────────────────────────────────────────────
// VulkanBackend — manages Vulkan device, queues, pipelines
// ─────────────────────────────────────────────

class VulkanBackend {
public:
    VulkanBackend(uint32_t gpu_index = 0) {
        init_instance();
        pick_physical_device(gpu_index);
        init_device();
        init_command_pool();
    }

    ~VulkanBackend() {
        cleanup();
    }

    // Prevent copying
    VulkanBackend(const VulkanBackend&) = delete;
    VulkanBackend& operator=(const VulkanBackend&) = delete;

    std::string device_name() const { return _device_name; }
    std::string api_version() const { return _api_version; }

    // ── Buffer management ────────────────────

    VulkanBuffer create_buffer(VkDeviceSize size, VkBufferUsageFlags usage) {
        VulkanBuffer buf;
        buf.size = size;

        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = size;
        bi.usage = usage;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(_device, &bi, nullptr, &buf.buffer));

        VkMemoryRequirements mem_req;
        vkGetBufferMemoryRequirements(_device, buf.buffer, &mem_req);

        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mem_req.size;
        ai.memoryTypeIndex = find_memory_type(mem_req.memoryTypeBits,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK(vkAllocateMemory(_device, &ai, nullptr, &buf.memory));
        VK_CHECK(vkBindBufferMemory(_device, buf.buffer, buf.memory, 0));

        return buf;
    }

    void* map_buffer(VulkanBuffer& buf) {
        if (!buf.mapped) {
            VK_CHECK(vkMapMemory(_device, buf.memory, 0, buf.size, 0, &buf.mapped));
        }
        return buf.mapped;
    }

    void unmap_buffer(VulkanBuffer& buf) {
        if (buf.mapped) {
            vkUnmapMemory(_device, buf.memory);
            buf.mapped = nullptr;
        }
    }

    void destroy_buffer(VulkanBuffer& buf) {
        if (buf.mapped) vkUnmapMemory(_device, buf.memory);
        if (buf.buffer) vkDestroyBuffer(_device, buf.buffer, nullptr);
        if (buf.memory) vkFreeMemory(_device, buf.memory, nullptr);
        buf = {};
    }

    // ── Compute shader execution ─────────────

    VkDescriptorSet create_descriptor_set(const std::vector<VkBuffer>& buffers,
                                           const std::vector<VkDeviceSize>& offsets,
                                           const void* push_constants = nullptr,
                                           uint32_t push_constants_size = 0) {
        // Create descriptor pool
        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = (uint32_t)buffers.size();

        VkDescriptorPoolCreateInfo pool_ci{};
        pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_ci.maxSets = 1;
        pool_ci.poolSizeCount = 1;
        pool_ci.pPoolSizes = &pool_size;
        VkDescriptorPool pool;
        VK_CHECK(vkCreateDescriptorPool(_device, &pool_ci, nullptr, &pool));

        // Create descriptor set layout
        std::vector<VkDescriptorSetLayoutBinding> bindings(buffers.size());
        for (uint32_t i = 0; i < buffers.size(); i++) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layout_ci{};
        layout_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_ci.bindingCount = (uint32_t)bindings.size();
        layout_ci.pBindings = bindings.data();
        VkDescriptorSetLayout layout;
        VK_CHECK(vkCreateDescriptorSetLayout(_device, &layout_ci, nullptr, &layout));

        // Allocate descriptor set
        VkDescriptorSetAllocateInfo alloc_ci{};
        alloc_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_ci.descriptorPool = pool;
        alloc_ci.descriptorSetCount = 1;
        alloc_ci.pSetLayouts = &layout;
        VkDescriptorSet desc_set;
        VK_CHECK(vkAllocateDescriptorSets(_device, &alloc_ci, &desc_set));

        // Update descriptor set
        std::vector<VkDescriptorBufferInfo> buf_infos(buffers.size());
        std::vector<VkWriteDescriptorSet> writes(buffers.size());
        for (uint32_t i = 0; i < buffers.size(); i++) {
            buf_infos[i].buffer = buffers[i];
            buf_infos[i].offset = offsets.empty() ? 0 : offsets[i];
            buf_infos[i].range = VK_WHOLE_SIZE;
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = desc_set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buf_infos[i];
        }
        vkUpdateDescriptorSets(_device, (uint32_t)writes.size(), writes.data(),
                               0, nullptr, 0, nullptr);

        return desc_set;
    }

    void dispatch(VkPipeline pipeline, VkPipelineLayout layout,
                  const std::vector<VkBuffer>& buffers,
                  const std::vector<VkDeviceSize>& offsets,
                  uint32_t group_x, uint32_t group_y = 1, uint32_t group_z = 1,
                  const void* push_constants = nullptr,
                  uint32_t push_constants_size = 0) {

        VkDescriptorSet desc_set = create_descriptor_set(buffers, offsets);

        VkCommandBuffer cmd = begin_command();

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout,
                                0, 1, &desc_set, 0, nullptr);

        if (push_constants && push_constants_size > 0) {
            vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, push_constants_size, push_constants);
        }

        vkCmdDispatch(cmd, group_x, group_y, group_z);

        end_command(cmd);
    }

    // ── High-level operations ────────────────

    void drelu(py::array_t<float> input, py::array_t<float> output, uint64_t n) {
        if (!_pipelines.count("drelu")) {
            _pipelines["drelu"] = create_pipeline(compile_spirv(DRELU_SPV));
        }

        auto buf_in = create_and_fill_buffer(input.data(), input.size());
        auto buf_out = create_buffer(output.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        void* out_ptr = map_buffer(buf_out);

        dispatch(_pipelines["drelu"], _pipeline_layout,
                 {buf_in.buffer, buf_out.buffer},
                 {}, (n + 255) / 256);

        std::memcpy(output.mutable_data(), out_ptr, n * sizeof(float));

        destroy_buffer(buf_in);
        destroy_buffer(buf_out);
    }

    void silu(py::array_t<float> input, py::array_t<float> output, uint64_t n) {
        if (!_pipelines.count("silu")) {
            _pipelines["silu"] = create_pipeline(compile_spirv(SILU_SPV));
        }

        auto buf_in = create_and_fill_buffer(input.data(), input.size());
        auto buf_out = create_buffer(output.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        void* out_ptr = map_buffer(buf_out);

        dispatch(_pipelines["silu"], _pipeline_layout,
                 {buf_in.buffer, buf_out.buffer},
                 {}, (n + 255) / 256);

        std::memcpy(output.mutable_data(), out_ptr, n * sizeof(float));

        destroy_buffer(buf_in);
        destroy_buffer(buf_out);
    }

    void matmul(py::array_t<float> a, py::array_t<float> b, py::array_t<float> c,
                uint64_t M, uint64_t K, uint64_t N) {
        if (!_pipelines.count("matmul")) {
            _pipelines["matmul"] = create_pipeline(compile_spirv(MATMUL_SPV));
        }

        struct { uint32_t M, K, N; } params = {(uint32_t)M, (uint32_t)K, (uint32_t)N};

        auto buf_a = create_and_fill_buffer(a.data(), a.size());
        auto buf_b = create_and_fill_buffer(b.data(), b.size());
        auto buf_c = create_buffer(c.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        void* c_ptr = map_buffer(buf_c);

        dispatch(_pipelines["matmul"], _pipeline_layout,
                 {buf_a.buffer, buf_b.buffer, buf_c.buffer},
                 {},
                 (M + 15) / 16, (N + 15) / 16, 1,
                 &params, sizeof(params));

        std::memcpy(c.mutable_data(), c_ptr, M * N * sizeof(float));

        destroy_buffer(buf_a);
        destroy_buffer(buf_b);
        destroy_buffer(buf_c);
    }

    void rms_norm(py::array_t<float> input, py::array_t<float> weight,
                  py::array_t<float> output, float eps, uint64_t dim) {
        if (!_pipelines.count("rms_norm")) {
            _pipelines["rms_norm"] = create_pipeline(compile_spirv(RMSNORM_SPV));
        }

        uint64_t rows = input.shape(0) / dim;
        struct { float eps; uint32_t dim; } params = {eps, (uint32_t)dim};

        auto buf_in = create_and_fill_buffer(input.data(), input.size());
        auto buf_w = create_and_fill_buffer(weight.data(), weight.size());
        auto buf_out = create_buffer(output.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        void* out_ptr = map_buffer(buf_out);

        dispatch(_pipelines["rms_norm"], _pipeline_layout,
                 {buf_in.buffer, buf_out.buffer, buf_w.buffer},
                 {}, rows, 1, 1, &params, sizeof(params));

        std::memcpy(output.mutable_data(), out_ptr, input.size());

        destroy_buffer(buf_in);
        destroy_buffer(buf_w);
        destroy_buffer(buf_out);
    }

private:
    VkInstance _instance = VK_NULL_HANDLE;
    VkPhysicalDevice _physical_device = VK_NULL_HANDLE;
    VkDevice _device = VK_NULL_HANDLE;
    VkQueue _queue = VK_NULL_HANDLE;
    uint32_t _queue_family = 0;
    VkCommandPool _command_pool = VK_NULL_HANDLE;
    VkPipelineLayout _pipeline_layout = VK_NULL_HANDLE;
    std::string _device_name;
    std::string _api_version;
    std::unordered_map<std::string, VkPipeline> _pipelines;

    // ── Initialization ──────────────────────

    void init_instance() {
        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "VibeBlade";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName = "VibeBlade";
        app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &app_info;

        VK_CHECK(vkCreateInstance(&ci, nullptr, &_instance));

        // Store API version
        uint32_t api_ver;
        vkEnumerateInstanceVersion(&api_ver);
        _api_version = std::to_string(VK_VERSION_MAJOR(api_ver)) + "." +
                       std::to_string(VK_VERSION_MINOR(api_ver)) + "." +
                       std::to_string(VK_VERSION_PATCH(api_ver));
    }

    void pick_physical_device(uint32_t index) {
        uint32_t count;
        vkEnumeratePhysicalDevices(_instance, &count, nullptr);
        if (count == 0) throw std::runtime_error("No Vulkan physical devices found");

        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(_instance, &count, devices.data());

        if (index >= count)
            throw std::runtime_error("GPU index " + std::to_string(index) +
                                     " out of range (found " + std::to_string(count) + ")");

        _physical_device = devices[index];

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(_physical_device, &props);
        _device_name = props.deviceName;

        // Find compute queue family
        uint32_t qf_count;
        vkGetPhysicalDeviceQueueFamilyProperties(_physical_device, &qf_count, nullptr);
        std::vector<VkQueueFamilyProperties> qf_props(qf_count);
        vkGetPhysicalDeviceQueueFamilyProperties(_physical_device, &qf_count, qf_props.data());

        for (uint32_t i = 0; i < qf_count; i++) {
            if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                _queue_family = i;
                break;
            }
        }
    }

    void init_device() {
        float queue_priority = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = _queue_family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &queue_priority;

        // Enable compute shader support
        const char* extensions[] = { "VK_KHR_shader_float16_int8" };
        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = extensions;

        VK_CHECK(vkCreateDevice(_physical_device, &dci, nullptr, &_device));
        vkGetDeviceQueue(_device, _queue_family, 0, &_queue);
    }

    void init_command_pool() {
        VkCommandPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = _queue_family;
        VK_CHECK(vkCreateCommandPool(_device, &ci, nullptr, &_command_pool));

        // Create pipeline layout (with push constants)
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = 128; // max push constant size

        VkPipelineLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        lci.pushConstantRangeCount = 1;
        lci.pPushConstantRanges = &pcr;
        VK_CHECK(vkCreatePipelineLayout(_device, &lci, nullptr, &_pipeline_layout));
    }

    // ── Shader compilation ──────────────────

    VkShaderModule compile_spirv(const std::vector<uint32_t>& spirv) {
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = spirv.size() * sizeof(uint32_t);
        ci.pCode = spirv.data();
        VkShaderModule module;
        VK_CHECK(vkCreateShaderModule(_device, &ci, nullptr, &module));
        return module;
    }

    VkPipeline create_pipeline(VkShaderModule shader) {
        VkComputePipelineCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = shader;
        ci.stage.pName = "main";
        ci.layout = _pipeline_layout;

        VkPipeline pipeline;
        VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &ci,
                                          nullptr, &pipeline));
        return pipeline;
    }

    // ── Command buffer helpers ───────────────

    VkCommandBuffer begin_command() {
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = _command_pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;

        VkCommandBuffer cmd;
        VK_CHECK(vkAllocateCommandBuffers(_device, &ai, &cmd));

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

        return cmd;
    }

    void end_command(VkCommandBuffer cmd) {
        VK_CHECK(vkEndCommandBuffer(cmd));

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;

        VkFence fence;
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VK_CHECK(vkCreateFence(_device, &fci, nullptr, &fence));

        VK_CHECK(vkQueueSubmit(_queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(_device, 1, &fence, VK_TRUE, 100000000000ULL));

        vkDestroyFence(_device, fence, nullptr);
        vkFreeCommandBuffers(_device, _command_pool, 1, &cmd);
    }

    // ── Memory helpers ──────────────────────

    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props) {
        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(_physical_device, &mem_props);
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
            if ((type_filter & (1u << i)) &&
                (mem_props.memoryTypes[i].propertyFlags & props) == props) {
                return i;
            }
        }
        throw std::runtime_error("Failed to find suitable memory type");
    }

    VulkanBuffer create_and_fill_buffer(const void* data, size_t size) {
        auto buf = create_buffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        void* ptr = map_buffer(buf);
        std::memcpy(ptr, data, size);
        return buf;
    }

    // ── Cleanup ─────────────────────────────

    void cleanup() {
        for (auto& [name, pipeline] : _pipelines) {
            vkDestroyPipeline(_device, pipeline, nullptr);
        }
        if (_pipeline_layout) vkDestroyPipelineLayout(_device, _pipeline_layout, nullptr);
        if (_command_pool) vkDestroyCommandPool(_device, _command_pool, nullptr);
        if (_device) vkDestroyDevice(_device, nullptr);
        if (_instance) vkDestroyInstance(_instance, nullptr);
    }

    // ── Embedded SPIR-V bytecodes ───────────
    // These would be pre-compiled from GLSL at build time.
    // For now, they're placeholders — actual SPIR-V is generated by CMake.

    static const std::vector<uint32_t> DRELU_SPV;     // compiled from kernels_vulkan.glsl
    static const std::vector<uint32_t> SILU_SPV;
    static const std::vector<uint32_t> MATMUL_SPV;
    static const std::vector<uint32_t> RMSNORM_SPV;
};

// Placeholder SPIR-V data — populated by CMake build from GLSL sources
const std::vector<uint32_t> VulkanBackend::DRELU_SPV = {};
const std::vector<uint32_t> VulkanBackend::SILU_SPV = {};
const std::vector<uint32_t> VulkanBackend::MATMUL_SPV = {};
const std::vector<uint32_t> VulkanBackend::RMSNORM_SPV = {};


// ─────────────────────────────────────────────
// Python module registration
// ─────────────────────────────────────────────

PYBIND11_MODULE(_vibeblade_vulkan, m) {
    m.doc() = "VibeBlade Vulkan cross-platform GPU backend";

    py::class_<VulkanBackend>(m, "VulkanBackend")
        .def(py::init<uint32_t>(), py::arg("gpu_index") = 0,
             "Initialize Vulkan backend")
        .def("device_name", &VulkanBackend::device_name,
             "Return GPU device name")
        .def("api_version", &VulkanBackend::api_version,
             "Return Vulkan API version")
        .def("drelu", &VulkanBackend::drelu,
             "dReLU activation sparsification",
             py::arg("input"), py::arg("output"), py::arg("n"))
        .def("silu", &VulkanBackend::silu,
             "SiLU activation (x * sigmoid(x))",
             py::arg("input"), py::arg("output"), py::arg("n"))
        .def("matmul", &VulkanBackend::matmul,
             "Matrix multiply C = A × B",
             py::arg("a"), py::arg("b"), py::arg("c"),
             py::arg("M"), py::arg("K"), py::arg("N"))
        .def("rms_norm", &VulkanBackend::rms_norm,
             "RMS normalization",
             py::arg("input"), py::arg("weight"), py::arg("output"),
             py::arg("eps"), py::arg("dim"));
}
