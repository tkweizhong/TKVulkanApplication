#define VK_USE_PLATFORM_WIN32_KHR


#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <iostream>
#include <stdexcept>
namespace std
{
#include <cstdlib>
}
#include <vector>
#include <optional>
#include <map>
#include <set>
#include <xstring>
#include "FileUtils.h"
#include "xstringextension.hpp"
#include <array>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
//using namespace std;

namespace TKVulkanNS
{
	/// <summary>
	/// 并不是每一个物理设备都支持窗体显示功能，我们需要检查物理设备是否可以图像呈现到我们创建的surface上;
	/// 支持graphics命令的队列簇 和  支持presentation命令的队列簇可能不是一个队列簇，因此这里需要分开两个字段来查询;
	/// 处于性能考虑，我们可以明确指定物理设备使用的graphics和presentation功能来自同一个队列簇;
	/// </summary>
	struct QueueFamilyIndices
	{
		int graphicsFamily = -1;
		int presentFamily = -1;

		bool isComplete()
		{
			return graphicsFamily >= 0 && presentFamily >= 0;
		}
	};

	struct Vertex
	{
		glm::vec2 pos;
		glm::vec3 color;

		static VkVertexInputBindingDescription getBindingDescription()
		{
			VkVertexInputBindingDescription bindingDescription = {};
			bindingDescription.binding = 0;
			bindingDescription.stride = sizeof(Vertex);
			bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
			return bindingDescription;
		}

		static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions()
		{
			std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions = {};
			attributeDescriptions[0].binding = 0;
			attributeDescriptions[0].location = 0;
			attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
			attributeDescriptions[0].offset = offsetof(Vertex, pos);
			attributeDescriptions[1].binding = 0;
			attributeDescriptions[1].location = 1;
			attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
			attributeDescriptions[1].offset = offsetof(Vertex, color);
			return attributeDescriptions;
		}
	};
	
	const std::vector<Vertex> vertices = {
		{{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
		{{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
		{{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
	};

	const std::vector<uint16_t> indices = {
		0,1,2,
		2,3,0
	};

	struct UniformBufferObject
	{
		glm::mat4 model;
		glm::mat4 view;
		glm::mat4 proj;
	};
	
	struct SwapChainSupportDetails
	{
		VkSurfaceCapabilitiesKHR capabilities;
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;
	};

	const uint32_t WINDOW_WIDTH = 800;
	const uint32_t WINDOW_HEIGHT = 600;

	const std::vector<const char*> validationLayers = {
		"VK_LAYER_KHRONOS_validation",
		//"VK_LAYER_LUNARG_api_dump"
	};

	const std::vector<const char*> deviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	};

	/// <summary>
	/// 通知回调函数
	/// </summary>
	/// <param name="flags"></param>
	/// <param name="objType"></param>
	/// <param name="obj"></param>
	/// <param name="location"></param>
	/// <param name="code"></param>
	/// <param name="layerPrefix"></param>
	/// <param name="msg"></param>
	/// <param name="userData"></param>
	/// <returns></returns>
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugReportFlagsEXT flags, //消息类型：VK_DEBUG_REPORT_INFORMATION_BIT_EXT xxxx_WARNING_xxx ...
		VkDebugReportObjectTypeEXT objType,//抛出该异常的对象类型，e.g.VkPhysicalDevice对应的VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_EXT 
		uint64_t obj,
		size_t location,
		int32_t code,
		const char* layerPrefix,
		const char* msg,
		void* userData
	){
		std::cerr << "validation layer : " << msg << std::endl;
		return VK_FALSE;
	}

	/// <summary>
	/// 创建debug通知回调
	/// </summary>
	/// <param name="instance"></param>
	/// <param name="pCreateInfo"></param>
	/// <param name="pAllocator"></param>
	/// <param name="pCallback"></param>
	/// <returns></returns>
	VkResult CreateDebugReportCallbackEXT(VkInstance instance,
		const VkDebugReportCallbackCreateInfoEXT* pCreateInfo,
		const VkAllocationCallbacks* pAllocator,
		VkDebugReportCallbackEXT* pCallback
		)
	{
		auto func = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
		if (func != nullptr)
		{
			return func(instance, pCreateInfo, pAllocator, pCallback);
		}
		else
		{
			return VK_ERROR_EXTENSION_NOT_PRESENT;
		}
	}
	/// <summary>
	/// 销毁debug通知回调;
	/// </summary>
	/// <param name="instance"></param>
	/// <param name="callback"></param>
	/// <param name="pAllocation"></param>
	void DestroyDebugReportCallbackEXT(VkInstance instance, const VkDebugReportCallbackEXT& callback, const VkAllocationCallbacks* pAllocation )
	{
		auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
		if (func != nullptr)
		{
			func(instance, callback, pAllocation);
		}
	}

	/// <summary>
	/// 选择最优绘制格式以及颜色空间;
	/// </summary>
	/// <param name="availableFormats"></param>
	/// <returns></returns>
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(std::vector<VkSurfaceFormatKHR>& availableFormats)
	{
		if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED)
		{
			return { VK_FORMAT_B8G8R8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
		} 
		for (const auto& availableFormat : availableFormats)
		{
			if (availableFormat.format == VK_FORMAT_B8G8R8_UNORM && availableFormat.colorSpace == VK_COLORSPACE_SRGB_NONLINEAR_KHR)
			{
				return availableFormat;
			}
		}
		return availableFormats[0];
	} 

	/// <summary>
	/// 选择最优绘制模式
	/// </summary>
	/// <param name="presentModes"></param>
	/// <returns></returns>
	VkPresentModeKHR chooseSwapChainPresentMode(std::vector<VkPresentModeKHR>& presentModes)
	{
		VkPresentModeKHR bestMode = VK_PRESENT_MODE_IMMEDIATE_KHR; //立即绘制模式;
		for (const auto& presentMode : presentModes)
		{
			switch (presentMode)
			{
			case VK_PRESENT_MODE_FIFO_KHR: //先入先出绘制模式，队列满时会等待;
				if (bestMode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
					bestMode = presentMode; 
				}
				break;
			case VK_PRESENT_MODE_FIFO_RELAXED_KHR://先入先出模式，队列空时会直接绘制到屏幕上;
				if (bestMode == VK_PRESENT_MODE_IMMEDIATE_KHR || bestMode == VK_PRESENT_MODE_FIFO_KHR)
					bestMode = presentMode; 
				break;
			case VK_PRESENT_MODE_MAILBOX_KHR: //先入先出模式，队列满时会替换旧的图像;可以比较有效的降低延迟带来的撕裂效果;
				return presentMode;
			default:
				break;
			}
		}
		return bestMode;
	}



#ifdef NDEBUG
	const bool enableValidationLayers = false;
#else
	const bool enableValidationLayers = true;
#endif

	class TKVulkanApplication
	{
	public:
		TKVulkanApplication();
		~TKVulkanApplication();

		void run();

	private:
		void initWindow();
		void createVulkanInstance();
		SwapChainSupportDetails querySwapChainSupport(const VkPhysicalDevice device);
		void createSurface();
		void initVulkan();
		void mainLoop();
		void cleanup();
		void pickPhsicDevice();
		bool isDeviceSuitable(const VkPhysicalDevice& device);
		int rateDeviceSuitableility(const VkPhysicalDevice& device);
		QueueFamilyIndices findQueueFamilies(const VkPhysicalDevice& physicalDevice);
		bool checkValidationLayerSupport();
		void setDebugCallback();

		void createLogicalDevice();
		void createSwapChain();
		void createImageViews();
		void createRenderPass();
		void createDescriptorPool();
		void createDescriptorSets();
		void createDescriptorSetLayout();
		void createGraphicsPipeline();
		void createFrameBuffers();
		void createCommandPool();
		void createTextureImage();
		void createTextureImageView();
		void createImageView();
		void createVertexBuffer();
		void createIndexBuffer();
		void createUniformBuffer();
		void updateUniformBuffer();
		void createCommandBuffers();
		void createSemaphores();
		void drawFrame();
		void recreateSwapChain();
		void cleanupSwapChain();
		VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
		void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
		void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
		void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
		void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
		bool hasStencilComponent(VkFormat format);

		VkCommandBuffer beginSingleTimeCommands();
		void endSingleTimeCommands(VkCommandBuffer commandBuffer);

		VkShaderModule createShaderModule(const std::vector<char>& code);
		bool checkDeviceExtensionSupport(const VkPhysicalDevice& physicalDevice);

		uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

		static void onWindowResized(GLFWwindow* window, int width, int height)
		{
			if (width == 0 || height == 0) return;
			auto app = reinterpret_cast<TKVulkanApplication*>(glfwGetWindowUserPointer(window));
			app->recreateSwapChain();
		}

	private:
		GLFWwindow* window;
		VkInstance vkInstance;
		VkDebugReportCallbackEXT callback;
		VkSurfaceKHR surface;
		VkPhysicalDevice physicalDevice;
		VkDevice device;
		VkQueue graphicsQueue;
		VkQueue presentQueue;
		VkSwapchainKHR swapChain;
		std::vector<VkImage> swapChainImages;
		std::vector<VkImageView> swapChainImageViews;

		VkSurfaceFormatKHR swapChainImageFormat;
		VkExtent2D swapChainExtent;

		VkPipelineLayout pipelineLayout;
		std::vector<VkFramebuffer> swapChainFramebuffers;

		VkPipeline graphicsPipeline;
		VkRenderPass renderPass;
		VkCommandPool commandPool;
		std::vector<VkCommandBuffer> commandBuffers;

		VkSemaphore imageAvailableSemaphore;
		VkSemaphore renderFinishedSemaphore;

		//VBO
		VkBuffer vertexBuffer;
		VkMemoryRequirements memRequirements;
		VkDeviceMemory vertexBufferMemory;
		//IBO
		VkBuffer indexBuffer;
		VkDeviceMemory indexBufferMemory;

		VkDescriptorSetLayout descriptorSetLayout;
		//VkPipelineLayout pipelineLayout;
		
		//UBO
		VkBuffer uniformBuffer;
		VkDeviceMemory uniformBufferMemory;

		VkDescriptorPool descriptorPool;
		VkDescriptorSet descriptorSet;

		//image
		VkImage textureImage;
		VkDeviceMemory textureImageMemory;
		VkImageView textureImageView;
	};

	TKVulkanApplication::TKVulkanApplication() = default;

	TKVulkanApplication::~TKVulkanApplication() = default;

	void TKVulkanApplication::run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}


	void TKVulkanApplication::initWindow()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "TK Vulkan Application", nullptr, nullptr);

		glfwSetWindowUserPointer(window, this);
		glfwSetWindowSizeCallback(window, TKVulkanApplication::onWindowResized);
	}

	/// <summary>
	/// 交换链图像的分辨率；绝大部分情况下这个分辨率就是窗口的大小;
	/// currentExtent的width/height 特殊值为unint32_t最大值时,允许自定义最佳交换链图形分辨率;
	/// </summary>
	/// <param name="capabilities"></param>
	/// <returns></returns>
	VkExtent2D TKVulkanApplication::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
	{
		if (capabilities.currentExtent.height != (std::numeric_limits<uint32_t>::max)())
		{
			return capabilities.currentExtent;
		}

		{
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);
			VkExtent2D actualExtent = { width, height };
			actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
			actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));
			return actualExtent;
		}
	}

	void PrintExtensions()
	{
		uint32_t extensionCount = 0;
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
		std::vector<VkExtensionProperties> extensionProperties(extensionCount);
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensionProperties.data());
		std::cout << "available extensions:\n";
		for (const auto& a : extensionProperties)
		{
			std::cout << "\t" << a.extensionName << "\n";
		}
	}

	bool CheckAllRequiredExtensionPropertiesLegal(const char** glfwExtensions = nullptr)
	{
		uint32_t glfwExtensionsCount = 0;
		if (glfwExtensions == nullptr)
			glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionsCount);

		std::vector<const char*> requiredExtensions;
		for (uint32_t i = 0; i < glfwExtensionsCount; ++i)
		{
			requiredExtensions.push_back(glfwExtensions[i]);
		}
		if (enableValidationLayers)
		{
			requiredExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
		}

		//ilegalExtensions structure is dummy, used for debug.
		std::vector<const char*> ilegalExtensions;

		uint32_t extensionsCount;
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionsCount, nullptr);
		std::vector<VkExtensionProperties> vkExtensions(extensionsCount);
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionsCount, vkExtensions.data());

		for (const auto glfwExtension: requiredExtensions)
		{
			auto finded = false;
			for (const auto& e : vkExtensions)
			{
				if (strcmp(glfwExtension, e.extensionName) == 0)
				{
					finded = true; break;
				}
			}
			if (!finded)
				ilegalExtensions.emplace_back(glfwExtension);
		}

		return ilegalExtensions.size() < 1;
	}

	bool TKVulkanApplication::checkValidationLayerSupport()
	{
		uint32_t layersCount;
		vkEnumerateInstanceLayerProperties(&layersCount, nullptr);
		std::vector<VkLayerProperties> availableLayers(layersCount);
		vkEnumerateInstanceLayerProperties(&layersCount, availableLayers.data());
		for (const auto& layerName : validationLayers)
		{
			bool layerFound = false;
			for (const auto& layerProperty : availableLayers)
			{
				if (strcmp(layerName, layerProperty.layerName) == 0)
				{
					layerFound = true; break;
				}
			}

			if (!layerFound)
			{
				return false;
			}
		}

		return true;
	}

	void TKVulkanApplication::createVulkanInstance()
	{
		if (enableValidationLayers && !checkValidationLayerSupport())
		{
			throw std::runtime_error("validation layers request,but not available");
		}

		//PrintExtensions();
		auto ilegal = CheckAllRequiredExtensionPropertiesLegal();
		if (!ilegal)
		{
			throw std::runtime_error("some glfwRequiredExtensions is illegal");
		}
		/// <summary>
		/// 应用信息;
		/// </summary>
		VkApplicationInfo vkApplicationInfo{};
		vkApplicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		vkApplicationInfo.pApplicationName = "TK Vulkan";
		vkApplicationInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 1);
		vkApplicationInfo.pEngineName = "TK Engine";
		vkApplicationInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		vkApplicationInfo.apiVersion = VK_API_VERSION_1_0;
		
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		VkInstanceCreateInfo vkInstanceCreateInfo{};
		vkInstanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		vkInstanceCreateInfo.pApplicationInfo = &vkApplicationInfo;
		if (enableValidationLayers)
		{
			vkInstanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			vkInstanceCreateInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else
		{
			vkInstanceCreateInfo.enabledLayerCount = 0;
		}

		std::vector<const char*> requiredExtensions;
		for (uint32_t i = 0; i < glfwExtensionCount; ++i)
		{
			requiredExtensions.emplace_back(glfwExtensions[i]);
		}

		if (enableValidationLayers)
		{
			requiredExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
		}

		//////////////////////////////////////////////////////////////////////////
		//如果使用MacOS Molten SDK,出现VK_ERROR_INCOMPATIBLE_DRIVER错误，可以尝试增加VK_KHR_PROTABILITY_ENUMERATION_EXTWNSION_NAME扩展;
		// 放开以下代码注释即可;
		/////////////////////2Ô/////////////////////////////////////////////////////
		//requiredExtensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
		//vkInstanceCreateInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
		vkInstanceCreateInfo.enabledExtensionCount = (uint32_t)requiredExtensions.size();
		vkInstanceCreateInfo.ppEnabledExtensionNames = requiredExtensions.data();

		VkResult result = vkCreateInstance(&vkInstanceCreateInfo, nullptr, &vkInstance);
		if (result != VK_SUCCESS)
		{
			const std::string log = std::format("failed to create vulkan instance! %d", result);
			throw std::runtime_error(log);
		}
	}

	void TKVulkanApplication::setDebugCallback()
	{
		if (!enableValidationLayers)
			return;
		VkDebugReportCallbackCreateInfoEXT createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
		createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_DEBUG_BIT_EXT;
		createInfo.pfnCallback = debugCallback;

		if (CreateDebugReportCallbackEXT(vkInstance, &createInfo, nullptr, &callback) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to set up callback!");
		}
	}

	void TKVulkanApplication::initVulkan()
	{
		createVulkanInstance();
		setDebugCallback();
		createSurface();
		pickPhsicDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();

		createRenderPass();
		createDescriptorSetLayout();
		createGraphicsPipeline();
		createFrameBuffers();
		createCommandPool();
		createTextureImage();
		createVertexBuffer();
		createIndexBuffer();
		createUniformBuffer();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBuffers();
	}

	void TKVulkanApplication::createLogicalDevice()
	{
		float queuePriority = 1.0f;

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		std::vector<VkDeviceQueueCreateInfo> deviceQueueCreateInfos;
		std::set<int> uniqueQueueFamilies = {indices.graphicsFamily, indices.presentFamily};
		for (const int queueFamily : uniqueQueueFamilies)
		{
			VkDeviceQueueCreateInfo queueCreateInfo = {};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;

			deviceQueueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures = {};

		VkDeviceCreateInfo deviceCreateInfo = {};
		deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
		deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(deviceQueueCreateInfos.size());
		deviceCreateInfo.pQueueCreateInfos = deviceQueueCreateInfos.data();
		deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
		if (enableValidationLayers) 
		{
			deviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			deviceCreateInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else
		{
			deviceCreateInfo.enabledLayerCount = 0;
		}

		if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create logical device!");
		}

		vkGetDeviceQueue(device, indices.graphicsFamily, 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily, 0, &presentQueue);
	}

	QueueFamilyIndices  TKVulkanApplication::findQueueFamilies(const VkPhysicalDevice& physicalDevice)
	{
		QueueFamilyIndices indices;
		uint32_t queueFamilyCount;
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

		for (int i=0; i< queueFamilies.size(); ++i)
		{
			const auto queueFamily = queueFamilies[i];
			if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				indices.graphicsFamily = i;
			}

			//检查设备是否支持呈现到surface;
			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);
			if (queueFamily.queueCount > 0 && presentSupport == VK_SUCCESS)
			{
				indices.presentFamily = i;
			}

			if (indices.isComplete())
				break;
		}

		return indices;
	}

	bool TKVulkanApplication::checkDeviceExtensionSupport(const VkPhysicalDevice& physicalDevice)
	{
		uint32_t extensionSupport = 0;
		vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionSupport, nullptr);
		std::vector<VkExtensionProperties> extensionProperties(extensionSupport);
		vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionSupport, extensionProperties.data());
		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto& extension : extensionProperties)
		{
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	bool TKVulkanApplication::isDeviceSuitable(const VkPhysicalDevice& device)
	{
		QueueFamilyIndices indices = findQueueFamilies(device);
		bool extensionsSupported = checkDeviceExtensionSupport(device);
		bool swapChainAdequate = false;
		if (extensionsSupported)
		{
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}
		return indices.isComplete() && extensionsSupported && swapChainAdequate;
	}

	int TKVulkanApplication::rateDeviceSuitableility(const VkPhysicalDevice& device)
	{
		int score = 0;

		VkPhysicalDeviceProperties physicalDeviceProperties;
		VkPhysicalDeviceFeatures physicalDeviceFeatures;
		vkGetPhysicalDeviceProperties(device, &physicalDeviceProperties);
		vkGetPhysicalDeviceFeatures(device, &physicalDeviceFeatures);

		if (physicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
			score += 1000;
		score += physicalDeviceProperties.limits.maxImageDimension2D;
		if (!physicalDeviceFeatures.geometryShader)
			return 0;
		return score;
	}

	void TKVulkanApplication::pickPhsicDevice()
	{
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(vkInstance, &deviceCount, nullptr);
		if (deviceCount == 0)
		{
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}
		std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
		vkEnumeratePhysicalDevices(vkInstance, &deviceCount, physicalDevices.data());

		std::multimap<int, VkPhysicalDevice> cadidates;
		for (auto& device : physicalDevices)
		{
			int score = rateDeviceSuitableility(device);
			cadidates.insert(std::pair(score, device));
		}
		const auto itor = cadidates.crbegin();
		if (itor!= cadidates.crend() && (*itor).first > 0)
		{
			physicalDevice = (*itor).second;
		}
		

		if (physicalDevice == VK_NULL_HANDLE)
		{
			throw std::runtime_error("failed to find GPU with Vulkan support!");
		}
	}

	SwapChainSupportDetails TKVulkanApplication::querySwapChainSupport(const VkPhysicalDevice device)
	{
		SwapChainSupportDetails details;
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
		uint32_t formatsCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatsCount, nullptr);
		if (formatsCount > 0)
		{
			details.formats.resize(formatsCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatsCount, details.formats.data());
		}
		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
		if (presentModeCount > 0)
		{
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}
		return details;
	}

	/// <summary>
	/// 创建surface有两种方式，
	/// 1，手动创建对应平台surface, 不同平台需要不同的创建方法和数据接口；
	/// 2，使用glfwCreateWindowSurface来兼容各平台接口;
	/// </summary>
	void TKVulkanApplication::createSurface()
	{
		//方式1 这里只会示意，加深对surface创建逻辑的理解;
		//VkWin32SurfaceCreateInfoKHR createInfo = {};
		//createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
		//createInfo.hwnd = glfwGetWin32Window(this->window);// hwnd:用于获取glfw窗口信息;
		//createInfo.hinstance = GetModuleHandle(nullptr);
		//auto CreateWin32SurfaceKHR = (PFN_vkCreateWin32SurfaceKHR)vkGetInstanceProcAddr(vkInstance, "vkCreateWin32SurfaceKHR");
		//if (!CreateWin32SurfaceKHR || CreateWin32SurfaceKHR(vkInstance, &createInfo, nullptr, &surface) != VK_SUCCESS)
		//{
		//	throw std::runtime_error("Create Win32 surface failed");
		//}

		//方式2 glfwCreateWindowSurface兼容不同平台创建窗口surface
		if (glfwCreateWindowSurface(vkInstance, window, nullptr, &surface) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create win32 surface");
		}
	}

	/// <summary>
	/// 创建交换链;
	/// </summary>
	void TKVulkanApplication::createSwapChain()
	{
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = chooseSwapChainPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		/// imageMaxCount等于0时，代表除了内存外,没有交换链图像队列的长度限制;
		uint32_t imageMaxCount = swapChainSupport.capabilities.maxImageCount;
		if (imageMaxCount > 0 && imageCount > imageMaxCount)
		{
			imageCount = imageMaxCount;
		}

		VkSwapchainCreateInfoKHR swapChainCreateInfo = {};
		swapChainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		swapChainCreateInfo.presentMode = presentMode;
		swapChainCreateInfo.imageExtent = extent;
		swapChainCreateInfo.imageColorSpace = surfaceFormat.colorSpace;
		swapChainCreateInfo.imageFormat = surfaceFormat.format;
		swapChainCreateInfo.minImageCount = imageCount;
		swapChainCreateInfo.surface = surface;
		swapChainCreateInfo.imageArrayLayers = 1;
		//指明图像用处;
		swapChainCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = {static_cast<uint32_t>(indices.graphicsFamily), static_cast<uint32_t>(indices.presentFamily)};
		if (indices.graphicsFamily != indices.presentFamily)
		{
			swapChainCreateInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			swapChainCreateInfo.queueFamilyIndexCount = 2;
			swapChainCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else
		{
			swapChainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			swapChainCreateInfo.queueFamilyIndexCount = 0;
			swapChainCreateInfo.pQueueFamilyIndices = nullptr;
		}
		//是否需要预旋转操作,e.g.90度顺时针旋转，水平翻转等;
		swapChainCreateInfo.preTransform = swapChainSupport.capabilities.currentTransform;
		//是否需要和其他窗体进行混合操作,这里默认使用非半透混合;
		swapChainCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		swapChainCreateInfo.clipped = VK_TRUE;
		//vulkan运行时，交换链在某些情况下需要被替换。比如窗口大小调整或者交换链重新分配更大的图像队列等。
		swapChainCreateInfo.oldSwapchain = VK_NULL_HANDLE;
		if (vkCreateSwapchainKHR(device, &swapChainCreateInfo, nullptr, &swapChain) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create swap chain");
		}

		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

		swapChainImageFormat = surfaceFormat;
		swapChainExtent = extent;
	}


	void TKVulkanApplication::createImageViews()
	{
		swapChainImageViews.resize(swapChainImages.size());
		for (size_t i = 0; i < swapChainImages.size(); ++i)
		{
			VkImageViewCreateInfo imageViewCreateInfo = {};
			imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			imageViewCreateInfo.image = swapChainImages[i];
			imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			imageViewCreateInfo.format = swapChainImageFormat.format;
			/// <summary>
			/// 指定View访问到的Image的通道映射，默认采用VK_COMPONENT_SWIZZLE_IDENTITY即可;
			/// VK_COMPONENT_SWIZZLE_IDENTITY 相当于imageViewCreateInfo.components.r/g/b/a = VK_COMPONENT_SWIZZLE_R/G/B/A;
			/// </summary>
			imageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			imageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			imageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			imageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

			/// <summary>
			/// aspectMask:指明view访问的是哪种类型的image
			/// baseArrayLayer/baseMipLevel: view能访问的image的起始ArrayLayer/Mipmap Level
			/// layerCount/levelCount: view能访问的ArrayLayer/mipmap数量，如果是从baseArrayLayer/baseMipLevel后所有的levels，
			/// 可以使用VK_REMAINING_ARRAY_LAYERS/VK_REMAINING_MIP_LEVELS 来处理
			/// </summary>
			imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
			imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
			imageViewCreateInfo.subresourceRange.layerCount = 1;
			imageViewCreateInfo.subresourceRange.levelCount = 1;

			if (vkCreateImageView(device, &imageViewCreateInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create image views");
			}
		}
	}

	void TKVulkanApplication::createRenderPass()
	{
		VkAttachmentDescription colorAttachment = {};
		colorAttachment.format = swapChainImageFormat.format;
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		/// <summary>
		/// layout:图像内存布局，cpu一般是线性布局，但gpu为了降低带宽使用的是tiled layout,
		/// 按照一定大小的tile排布的；
		/// VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIONAL 图像作为颜色附件
		/// VK_IMAGE_LAYOUT_PRESENT_SRC_KHR 图像作为交换链被呈现
		/// VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL 图像作为目标，用于内存COPY操作
		/// </summary>
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		//sub pass
		VkAttachmentReference colorAttachmentRef = {};
		colorAttachmentRef.attachment = 0;
		//colorAttachmentRef.layout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpassDes = {};
		subpassDes.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpassDes.colorAttachmentCount = 1;
		subpassDes.pColorAttachments = &colorAttachmentRef;

		VkRenderPassCreateInfo renderPassCreateInfo = {};
		renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassCreateInfo.attachmentCount = 1;
		renderPassCreateInfo.pAttachments = &colorAttachment;
		renderPassCreateInfo.subpassCount = 1;
		renderPassCreateInfo.pSubpasses = &subpassDes;

		/// <summary>
		/// 指定subpass的依赖关系，防止出现在管线开始时布局切换，但图像还没真正获取；
		/// 可以通过imageAvailableSemaphore的waitStages修改为VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,确保图像有效之前，渲染通道不会开始；
		/// 也可以渲染通道等到VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT；
		/// </summary>
		VkSubpassDependency dependency = {};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		renderPassCreateInfo.dependencyCount = 1;
		renderPassCreateInfo.pDependencies = &dependency;
		if (vkCreateRenderPass(device, &renderPassCreateInfo, nullptr, &renderPass) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create render pass");
		}
	}

	void TKVulkanApplication::createDescriptorSets()
	{
		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = 1;
		VkDescriptorSetLayout layouts[] = { descriptorSetLayout };
		allocInfo.pSetLayouts = layouts;

		if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate descriptor set");
		}

		VkDescriptorBufferInfo bufferInfo = {};
		bufferInfo.buffer = uniformBuffer;
		bufferInfo.offset = 0;
		bufferInfo.range = sizeof(UniformBufferObject);

		VkWriteDescriptorSet descriptorWrite = {};
		descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrite.dstSet = descriptorSet;
		descriptorWrite.dstBinding = 0;
		descriptorWrite.dstArrayElement = 0;
		descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorWrite.descriptorCount = 1;
		descriptorWrite.pBufferInfo = &bufferInfo;
		descriptorWrite.pImageInfo = nullptr;
		descriptorWrite.pTexelBufferView = nullptr;

		vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
	}

	void TKVulkanApplication::createDescriptorPool()
	{
		VkDescriptorPoolSize poolSize = {};
		poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSize.descriptorCount = 1;

		VkDescriptorPoolCreateInfo poolCreateInfo = {};
		poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolCreateInfo.poolSizeCount = 1;
		poolCreateInfo.pPoolSizes = &poolSize;
		poolCreateInfo.maxSets = 1;

		if (vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &descriptorPool) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create descriptor pool");
		}
	}

	void TKVulkanApplication::createDescriptorSetLayout()
	{
		VkDescriptorSetLayoutBinding uboLayoutBinding = {};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.pImmutableSamplers = nullptr;
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutCreateInfo layoutCreateInfo = {};
		layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutCreateInfo.bindingCount = 1;
		layoutCreateInfo.pBindings = &uboLayoutBinding;

		if (vkCreateDescriptorSetLayout(device, &layoutCreateInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create descriptor set layout!");
		}
	}

	void TKVulkanApplication::createGraphicsPipeline()
	{
		/// <summary>
		/// Shader Stage
		/// </summary>
		auto vertShaderCode = readFile("Shaders/vert.spv");
		auto fragShaderCode = readFile("Shaders/frag.spv");
		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		/// <summary>
		/// Vertex input state
		/// </summary>
		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();
		VkPipelineVertexInputStateCreateInfo vertexInputStageCreateInfo = {};
		vertexInputStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputStageCreateInfo.vertexBindingDescriptionCount = 1;
		vertexInputStageCreateInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputStageCreateInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputStageCreateInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		/// <summary>
		/// Assembly State
		/// </summary>
		VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;//三点成面，顶点不共用;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		/// <summary>
		/// ViewPort State;
		/// </summary>
		VkViewport viewPort = {};
		viewPort.x = 0; viewPort.y = 0;
		viewPort.width = (float)swapChainExtent.width;
		viewPort.height = (float)swapChainExtent.height;
		viewPort.minDepth = 0;
		viewPort.maxDepth = 1.f;

		VkRect2D scissor = {};
		scissor.extent = swapChainExtent;
		scissor.offset = { 0, 0 };

		VkPipelineViewportStateCreateInfo viewportStateCreateInfo = {};
		viewportStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportStateCreateInfo.pScissors = &scissor;
		viewportStateCreateInfo.pViewports = &viewPort;
		viewportStateCreateInfo.scissorCount = 1;
		viewportStateCreateInfo.viewportCount = 1;
		//viewportStateCreateInfo.flags = 
		
		VkPipelineRasterizationStateCreateInfo rastaerizationCreateInfo = {};
		rastaerizationCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		//depthClampEnable 设置为VK_TRUE 超过远近裁剪面的片元会进行收敛，而不是直接丢弃他们;
		//在特殊情况下比较有用，渲染级联阴影贴图时;
		rastaerizationCreateInfo.depthClampEnable = VK_FALSE;
		//rasterizerDiscardEnable设置为VK_TRUE,图元永远不会传递到光栅化阶段，可以禁止任何输出到framebuffer;
		rastaerizationCreateInfo.rasterizerDiscardEnable = VK_FALSE;
		rastaerizationCreateInfo.polygonMode = VK_POLYGON_MODE_FILL;//多边形区域填充
		//最大线宽取决于硬件，任何大于1.0的线宽都需要开启GPU wideLines特性支持;
		rastaerizationCreateInfo.lineWidth = 1;
		rastaerizationCreateInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rastaerizationCreateInfo.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rastaerizationCreateInfo.depthBiasEnable = VK_FALSE;
		rastaerizationCreateInfo.depthBiasConstantFactor = 0.f;
		rastaerizationCreateInfo.depthBiasClamp = 0.f;
		rastaerizationCreateInfo.depthBiasSlopeFactor = 0.f;

		//msaa state
		VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo = {};
		multisampleStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampleStateCreateInfo.sampleShadingEnable = VK_FALSE;
		multisampleStateCreateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampleStateCreateInfo.minSampleShading = 1.0f;
		multisampleStateCreateInfo.pSampleMask = nullptr;
		multisampleStateCreateInfo.alphaToCoverageEnable = VK_FALSE;
		multisampleStateCreateInfo.alphaToOneEnable = VK_FALSE;

		//Blend State
		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;

		VkPipelineColorBlendStateCreateInfo colorBlendStateCreateInfo = {};
		colorBlendStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlendStateCreateInfo.attachmentCount = 1;
		colorBlendStateCreateInfo.pAttachments = &colorBlendAttachment;
		colorBlendStateCreateInfo.logicOpEnable = VK_FALSE;
		colorBlendStateCreateInfo.logicOp = VK_LOGIC_OP_COPY;
		colorBlendStateCreateInfo.blendConstants[0] = 0;//r blend constant factor
		colorBlendStateCreateInfo.blendConstants[1] = 0;//g blend constant factor
		colorBlendStateCreateInfo.blendConstants[2] = 0;//b blend constant factor
		colorBlendStateCreateInfo.blendConstants[3] = 0;//a blend constant factor

		//Dynamic State
		VkDynamicState dynamicStates[] = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_LINE_WIDTH,
		};
		VkPipelineDynamicStateCreateInfo dynamicStateCreateInfo = {};
		dynamicStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicStateCreateInfo.dynamicStateCount = 2;
		dynamicStateCreateInfo.pDynamicStates = dynamicStates;
		
		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
		pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutCreateInfo.setLayoutCount = 1;
		pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
		pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
		pipelineLayoutCreateInfo.pPushConstantRanges = nullptr;

		if (vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create pipeline layout");
		}

		VkGraphicsPipelineCreateInfo  graphicsPipelineCreateInfo = {};
		graphicsPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		graphicsPipelineCreateInfo.stageCount = 2;
		graphicsPipelineCreateInfo.pStages = shaderStages;
		graphicsPipelineCreateInfo.pVertexInputState = &vertexInputStageCreateInfo;
		graphicsPipelineCreateInfo.pViewportState = &viewportStateCreateInfo;
		graphicsPipelineCreateInfo.pRasterizationState = &rastaerizationCreateInfo;
		graphicsPipelineCreateInfo.pMultisampleState = &multisampleStateCreateInfo;
		graphicsPipelineCreateInfo.pInputAssemblyState = &inputAssembly;
		graphicsPipelineCreateInfo.pDepthStencilState = nullptr;
		graphicsPipelineCreateInfo.pColorBlendState = &colorBlendStateCreateInfo;
		graphicsPipelineCreateInfo.pDynamicState = nullptr;// &dynamicStateCreateInfo;
		graphicsPipelineCreateInfo.layout = pipelineLayout;
		graphicsPipelineCreateInfo.renderPass = renderPass;
		graphicsPipelineCreateInfo.subpass = 0;
		graphicsPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
		graphicsPipelineCreateInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &graphicsPipelineCreateInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create graphics pipelines");
		}
	}

	VkShaderModule TKVulkanApplication::createShaderModule(const std::vector<char>& code)
	{
		VkShaderModuleCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
		VkShaderModule shaderModule;
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create shader module");
		}
		return shaderModule;
	}

	void TKVulkanApplication::createFrameBuffers()
	{
		swapChainFramebuffers.resize( swapChainImageViews.size() );
		for (size_t i = 0; i < swapChainImageViews.size(); ++i)
		{
			VkImageView attachments[] = {
				swapChainImageViews[i]
			};
			
			VkFramebufferCreateInfo frambufferCreateIfo = {};
			frambufferCreateIfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			frambufferCreateIfo.attachmentCount = 1;
			frambufferCreateIfo.pAttachments = attachments;
			frambufferCreateIfo.renderPass = renderPass;
			frambufferCreateIfo.width = swapChainExtent.width;
			frambufferCreateIfo.height = swapChainExtent.height;
			frambufferCreateIfo.layers = 1;

			if (vkCreateFramebuffer(device, &frambufferCreateIfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create framebuffer");
			}
		}
	}

	void TKVulkanApplication::createCommandPool()
	{
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		VkCommandPoolCreateInfo commandPoolCreateInfo = {};
		commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;
		/// <summary>
		/// VK_COMMAND_POOL_CREATE_TRANSIENT_BIT 提示命令缓冲区非常频繁的重新记录新命令,一般在帧结束时重置或者释放;
		/// VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT 允许命令缓冲区单独重置，没有这个标志，所有的缓冲区都必须一起重置; vkResetCommandBuffer主动重置，或者vkBeginCommandBuffer时隐士重置;
		///	VK_COMMAND_POOL_CREATE_PROTECTED_BIT 
		/// </summary>
		commandPoolCreateInfo.flags = 0;

		if (vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &commandPool) != VK_SUCCESS)
		{
			throw std::runtime_error("create command pool failed");
		}
	}

	void TKVulkanApplication::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferCopy copyRegion = {};
		copyRegion.size = size;
		copyRegion.srcOffset = 0;
		copyRegion.srcOffset = 0;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		endSingleTimeCommands(commandBuffer);
	}

	void TKVulkanApplication::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
	{
		VkBufferCreateInfo bufferCreateInfo = {};
		bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferCreateInfo.size = size;
		bufferCreateInfo.usage = usage;
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create buffer");
		}

		VkMemoryRequirements memoryRequirements;
		vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);

		VkMemoryAllocateInfo allocateInfo = {};
		allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocateInfo.allocationSize = memoryRequirements.size;
		allocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, properties);

		///实际生产环境，不要使用vkAllocateMemory为每一个CommandBuffer分配内存，这样才效率太低；定制一个内存分配器，采用内存分配器分配一个大块内存，使用offset偏移索引，并复用；
		///第二个原因是内存分配受到maxMemoryAllocationCount物理设备所限，GTX1080也只能分配4096的大小;
		if (vkAllocateMemory(device, &allocateInfo, nullptr, &bufferMemory) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate buffer memory");
		}

		vkBindBufferMemory(device, buffer, bufferMemory, 0);
	}

	void TKVulkanApplication::copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();
			
		VkBufferImageCopy region = {};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;
		region.imageOffset = { 0,0,0 };
		region.imageExtent = { width, height, 1 };
		vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		endSingleTimeCommands(commandBuffer);
	}

	bool TKVulkanApplication::hasStencilComponent(VkFormat format)
	{
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D16_UNORM_S8_UINT;
	}

	void TKVulkanApplication::transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageMemoryBarrier barrier = {};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;
		if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
		{
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
			if (hasStencilComponent(format))
			{
				barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
			}
		}
		else
		{
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		VkPipelineStageFlags sourceStage;
		VkPipelineStageFlags destinationStage;
		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
		{
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
		{
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
		{
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
		{
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		}
		else
		{
			throw std::invalid_argument("unsupported layout transition");
		}
		vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 
			0, nullptr, 0, nullptr, 1, &barrier);

		endSingleTimeCommands(commandBuffer);
	}

	void TKVulkanApplication::createTextureImage()
	{
		int texWidth, texHeight, texChannels;
		stbi_uc* pixels = stbi_load("Shaders/textures/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		VkDeviceSize imageSize = texWidth * texHeight * 4;
		if (!pixels)
		{
			throw std::runtime_error("failed to load texture image");
		}

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer,  stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
		memcpy(data, pixels, static_cast<size_t>(imageSize));
		vkUnmapMemory(device, stagingBufferMemory);

		stbi_image_free(pixels);

		VkImageCreateInfo imageCreateInfo = {};
		imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
		imageCreateInfo.extent.width  = static_cast<uint32_t>(texWidth);
		imageCreateInfo.extent.height = static_cast<uint32_t>(texHeight);
		imageCreateInfo.extent.depth = 1;
		imageCreateInfo.mipLevels = 1;
		imageCreateInfo.arrayLayers = 1;
		imageCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
		imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCreateInfo.flags = 0;

		if (vkCreateImage(device, &imageCreateInfo, nullptr, &textureImage) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create image");
		}

		VkMemoryRequirements memoryRequirements;
		vkGetImageMemoryRequirements(device, textureImage, &memoryRequirements);

		VkMemoryAllocateInfo allocateInfo = {};
		allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocateInfo.allocationSize = memoryRequirements.size;
		allocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		if (vkAllocateMemory(device, &allocateInfo, nullptr, &textureImageMemory) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate image memory");
		}

		vkBindImageMemory(device, textureImage, textureImageMemory, 0);
		
		transitionImageLayout(textureImage, VK_FORMAT_R8G8B8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void TKVulkanApplication::createTextureImageView()
	{
		VkImageViewCreateInfo viewCreateInfo = {};
		viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewCreateInfo.image = textureImage;
		viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
		viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewCreateInfo.subresourceRange.baseMipLevel = 0;
		viewCreateInfo.subresourceRange.levelCount = 1;
		viewCreateInfo.subresourceRange.baseArrayLayer = 0;
		viewCreateInfo.subresourceRange.layerCount = 1;
		
		if (vkCreateImageView(device, &viewCreateInfo, nullptr, &textureImageView) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create image view");
		}
	}

	void TKVulkanApplication::createVertexBuffer()
	{
		VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		//填充顶点缓存区
		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, vertices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

		copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void TKVulkanApplication::createIndexBuffer()
	{
		VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, indices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

		copyBuffer(stagingBuffer, indexBuffer, bufferSize);
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void TKVulkanApplication::createUniformBuffer()
	{
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);
		createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffer, uniformBufferMemory);
	}

	void TKVulkanApplication::createCommandBuffers()
	{
		commandBuffers.resize(swapChainFramebuffers.size());

		VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
		commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		commandBufferAllocateInfo.commandBufferCount = (uint32_t)commandBuffers.size();
		commandBufferAllocateInfo.commandPool = commandPool;
		/// <summary>
		/// VK_COMMAND_BUFFER_LEVEL_PRIMARY 可以提交到队列执行，但不能从其他的命令缓冲区调用;
		/// VK_COMMAND_BUFFER_LEVEL_SECONDARY 无法直接提交，但可以从主命令缓冲区调用;
		/// </summary>
		commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		
		if (vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, commandBuffers.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create command buffers");
		}

		for (size_t i = 0; i < commandBuffers.size(); ++i)
		{
			VkCommandBufferBeginInfo beginInfo = {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			/// <summary>
			/// VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT commandBuffer执行一次后立即清理；
			/// VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT 辅助型commandBuffer，限制在一个renderpass中生效;
			/// VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT commandBuffer可以重复提交，同时也在等待执行;
			/// </summary>
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
			beginInfo.pInheritanceInfo = nullptr;

			vkBeginCommandBuffer(commandBuffers[i], &beginInfo);
		}

		for (size_t i =0; i<commandBuffers.size(); ++i)
		{
			VkRenderPassBeginInfo renderPassBeginInfo = {};
			renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassBeginInfo.renderPass = renderPass;
			renderPassBeginInfo.framebuffer = swapChainFramebuffers[i];
			renderPassBeginInfo.renderArea.extent = swapChainExtent;
			renderPassBeginInfo.renderArea.offset = { 0, 0 };
			renderPassBeginInfo.clearValueCount = 1;
			VkClearValue clearColor = {0, 0, 0, 0};
			renderPassBeginInfo.pClearValues = &clearColor;

			/// <summary>
			/// VK_SUBPASS_CONTENTS_INLINE 渲染过程命令被嵌入在主命令缓冲区中，没有辅助缓冲区执行;
			/// VK_SUBPASS_CONTENTS_SECONDARY_COOMAND_BUFFERS 渲染通道命令将会从辅助命令缓冲区执行;
			/// </summary>
			vkCmdBeginRenderPass(commandBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
			vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

			VkBuffer vertexBuffers[] = { vertexBuffer };
			VkDeviceSize offsets[] = { 0 };
			vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);
			vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT16);
			vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
			vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
			//vkCmdDraw(commandBuffers[i], static_cast<uint32_t>(vertices.size()), 1, 0, 0);
			vkCmdEndRenderPass(commandBuffers[i]);
			if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to record command buffer!");
			}
		}
	}

	/// <summary>
	/// 创建信号量;
	/// </summary>
	void TKVulkanApplication::createSemaphores()
	{
		VkSemaphoreCreateInfo semaphoreCreateInfo = {};
		semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		auto success = vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &imageAvailableSemaphore) == VK_SUCCESS;
		success = success && vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &renderFinishedSemaphore) == VK_SUCCESS;
		if (!success)
		{
			throw std::runtime_error("failed to create semaphore");
		}
	}

	/// <summary>
	/// Fence:栅栏 主要用于应用程序和gpu渲染操作进行同步；
	/// 信号量：用于命令队列内或者跨命令队列同步操作
	/// subpass:会自动处理布局的转换
	/// </summary>
	void TKVulkanApplication::drawFrame()
	{
		vkQueueWaitIdle(presentQueue);
		createSemaphores();
		/// <summary>
		/// 从交换链换取图像;
		/// </summary>
		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(device, swapChain, (std::numeric_limits<uint64_t>::max)(), imageAvailableSemaphore,
			VK_NULL_HANDLE, &imageIndex);
		//交换链与surface 不再兼容，不可进行渲染.
		if (result == VK_ERROR_OUT_OF_DATE_KHR)
		{
			recreateSwapChain();
			return;
		}
		//交换链仍然可以向surface提交图像，但是surface的属性不再匹配正确。比如平台可能重新调整图像的尺寸适应窗体大小;
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
		{
			throw std::runtime_error("failed to acquire swap chain image");
		}

		/// <summary>
		/// 提交命令缓冲区;
		/// </summary>
		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		
		VkSemaphore waitSemaphores[] = {imageAvailableSemaphore};
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT  };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;
		
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

		VkSemaphore signalSemaphores[] = {renderFinishedSemaphore};
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to submit draw command buffer");
		}

		//呈现图像，提交到交换链显示到屏幕
		VkPresentInfoKHR presentInfo = {};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;
		VkSwapchainKHR swapChains[] = { swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;
		presentInfo.pResults = nullptr;

		result = vkQueuePresentKHR(presentQueue, &presentInfo);
		//如果是非最佳状态，也重新创建交换链。确保效果没有任何差错，尝试调整窗体大小，帧缓冲区大小变化与窗体匹配.
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
		{
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS)
		{
			throw std::runtime_error("failed to present swap chain image");
		}
		vkQueueWaitIdle(presentQueue);
	}

	void TKVulkanApplication::updateUniformBuffer()
	{
		static auto startTime = std::chrono::high_resolution_clock::now();
		static auto currentTime = std::chrono::high_resolution_clock::now();

		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count() / 1000.0f;
		UniformBufferObject ubo = {};
		ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);
		ubo.proj[1][1] *= -1;

		void* data;
		vkMapMemory(device, uniformBufferMemory, 0, sizeof(ubo), 0, &data);
		memcpy(data, &ubo, sizeof(ubo));
		vkUnmapMemory(device, uniformBufferMemory);
	}

	void TKVulkanApplication::recreateSwapChain()
	{
		vkDeviceWaitIdle(device);

		createSwapChain();
		createImageViews();
		createRenderPass();
		createGraphicsPipeline();
		createFrameBuffers();
		createCommandBuffers();
	}

	void TKVulkanApplication::cleanupSwapChain()
	{
		for (size_t i = 0; i < swapChainFramebuffers.size(); i++)
		{
			vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);
		}

		vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());

		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyRenderPass(device, renderPass, nullptr);
		for (size_t i = 0; i<swapChainImageViews.size(); ++i)
		{
			vkDestroyImageView(device, swapChainImageViews[i], nullptr);
		}
		vkDestroySwapchainKHR(device, swapChain, nullptr);
	}


	VkCommandBuffer TKVulkanApplication::beginSingleTimeCommands()
	{
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = 1;
		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(commandBuffer, &beginInfo);
		return commandBuffer;
	}

	void TKVulkanApplication::endSingleTimeCommands(VkCommandBuffer commandBuffer)
	{
		vkEndCommandBuffer(commandBuffer);
		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicsQueue);
		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	}


	/// <summary>
	/// 
	/// </summary>
	/// <param name="typeFilter"></param>
	/// <param name="properties"></param>
	/// <returns></returns>
	uint32_t TKVulkanApplication::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
		{
			if ((typeFilter & (1 << i)) 
				&& (memProperties.memoryTypes[i].propertyFlags & properties) == properties
				)
			{
				return i;
			}
		}
		throw std::runtime_error("failed to find suitable memory type!");
	}

	void TKVulkanApplication::mainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
			updateUniformBuffer();
			drawFrame();
		}

		/// <summary>
		///等待设备完成当前渲染操作空闲时，渲染相关操作还在进行，出现各种报错;
		/// </summary>
		vkDeviceWaitIdle(device);
	}

	void TKVulkanApplication::cleanup()
	{
		cleanupSwapChain();
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
		//vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

		vkDestroyBuffer(device, vertexBuffer, nullptr);
		vkDestroyImage(device, textureImage, nullptr);
		vkFreeMemory(device, vertexBufferMemory, nullptr);
		vkDestroyBuffer(device, indexBuffer, nullptr);
		vkFreeMemory(device, indexBufferMemory, nullptr);
		vkDestroyBuffer(device, uniformBuffer, nullptr);
		vkFreeMemory(device, uniformBufferMemory, nullptr);

		if (enableValidationLayers)
			DestroyDebugReportCallbackEXT(vkInstance, callback, nullptr);

		vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);
		vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);

		vkDestroyCommandPool(device, commandPool, nullptr);
		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		//vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyRenderPass(device, renderPass, nullptr);
		for (size_t i = 0; i < swapChainImageViews.size(); ++i)
		{
			vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);
		}
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		for (size_t i = 0; i < swapChainImageViews.size(); ++i)
		{
			vkDestroyImageView(device, swapChainImageViews[i], nullptr);
		}
		vkDestroySwapchainKHR(device, swapChain, nullptr);
		vkDestroyDevice(device, nullptr);
		vkDestroySurfaceKHR(vkInstance, surface, nullptr);
		vkDestroyInstance(vkInstance, nullptr);
		vkInstance = nullptr;
		glfwDestroyWindow(window);
		glfwTerminate();
		window = nullptr;
	}
}

int main()
{
	TKVulkanNS::TKVulkanApplication app;
	try
	{
		app.run();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}