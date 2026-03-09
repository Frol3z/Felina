#include "Application.hpp"

#include "ResourceManager.hpp"
#include "Window.hpp"
#include "UI.hpp"
#include "Scene.hpp"
#include "Renderer.hpp"
#include "GltfLoader.hpp"
#include "Input.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

namespace Felina
{
	Application::Application(const std::string& name, uint32_t windowWidth, uint32_t windowHeight)
		: m_name(name), m_startupWindowWidth(windowWidth), m_startupWindowHeight(windowHeight)
	{
	}

	Application::~Application() = default;

	void Application::Init()
	{
		LOG("[Application] Initializing...");

		InitGlfw();
		m_window = std::make_unique<Window>(m_startupWindowWidth, m_startupWindowHeight, m_name);
		glfwSetWindowUserPointer(m_window->GetHandle(), this);
		glfwSetFramebufferSizeCallback(m_window->GetHandle(), Application::FramebufferResizeCallback);
		glfwSetScrollCallback(m_window->GetHandle(), Application::ScrollCallback);

		m_input = std::make_unique<Input>();
		m_UI = std::make_unique<UI>();
		m_scene = std::make_unique<Scene>(static_cast<float>(m_startupWindowWidth), static_cast<float>(m_startupWindowHeight));
		m_renderer = std::make_unique<Renderer>(*this, *m_window, *m_scene);

		// Load default material
		std::unique_ptr<Material> defaultMaterial = std::make_unique<Material>(
			glm::vec3(0.5f), // Base color (light gray)
			glm::vec4(0.0f, 1.0f, 0.0f, 0.0f) // Metallic-roughness (100% rough)
		);
		auto& rm = ResourceManager::GetInstance();
		MaterialID defaultMaterialId = rm.LoadMaterial(std::move(defaultMaterial), "Default Material");
		rm.SetDefaultMaterial(defaultMaterialId);
		LOG("[Application] Loading default material...");

		InitImGui();
		LoadScene();

		LOG("[Application] Done.");
	}

	void Application::Run()
	{
		// MAIN LOOP
		while (!m_window->ShouldClose()) {
			glfwPollEvents();
			Update();
			m_renderer->DrawFrame();
		}

		LOG("[Application] Waiting for pending GPU operations to finish...");
		m_renderer->WaitIdle();
	}

	void Application::CleanUp()
	{
		LOG("[Application] Cleaning up...");

		// Unload all resources
		LOG("[Application] Unloading resources...");
		auto& rm = ResourceManager::GetInstance();
		rm.UnloadAll();

		// ImGui
		LOG("[Application] Destroying DearImGui context...");
		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();

		// GLFW
		LOG("[Application] Destroying the window...");
		m_window.reset();
		glfwTerminate();

		LOG("[Application] Done.");
	}

	void Application::FramebufferResizeCallback(GLFWwindow* window, int width, int height)
	{
		Application* app = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
		assert(app != nullptr && "[Application] glfwGetWindowUserPointer hasn't been set.");

		// Update window
		Window& win = app->GetWindow();
		win.SetWidth(static_cast<uint32_t>(width));
		win.SetHeight(static_cast<uint32_t>(height));

		// Update camera viewport
		Camera& camera = app->GetScene().GetCamera();
		camera.UpdateProjectionMatrix(static_cast<float>(width), static_cast<float>(height));

		// Signal the renderer
		app->SignalFramebufferResized();
	}

	// For a classic vertical mouse-wheel xOffset should be ignored 
	void Application::ScrollCallback(GLFWwindow* window, double xOffset, double yOffset)
	{
		Application* app = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
		assert(app != nullptr && "[Application] glfwGetWindowUserPointer hasn't been set.");

		app->GetInput().SetMouseScroll(0.35f * yOffset); // WIP
	}

	void Application::InitGlfw() 
	{
		glfwInit();
	}

	// NOTE: the renderer must have been initialized before calling this function
	void Application::InitImGui()
	{
		ImGui::CreateContext();

		ImGui::StyleColorsDark();
		ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;

		// Init backends
		ImGui_ImplGlfw_InitForVulkan(m_window->GetHandle(), true); // Second parameter is for chaining input callbacks to ImGui automatically
		ImGui_ImplVulkan_InitInfo vkInitInfo = m_renderer->GetImGuiInitInfo();
		ImGui_ImplVulkan_Init(&vkInitInfo);
	}

	void Application::LoadScene(const std::filesystem::path& filepath)
	{	
		// Wait for GPU operations to finish
		m_renderer->WaitIdle();

		// Unload previous resources (if a scene was already loaded)
		m_scene->ClearObjects();
		// TODO:
		// if multiple resources should be preserved a more robust
		// method should be implemented
		ResourceManager::GetInstance().UnloadSceneResources(m_renderer->GetSkyboxCubemapId());

		// TODO: I can avoid reloading the skybox each time I reload the scene
		LOG("[Application] Loading skybox...");
		m_renderer->LoadSkybox(SKYBOX);
		LOG("[Application] Skybox loaded successfully!");

		LOG("[Application] Loading scene from " + filepath.string() + "...");
		
		LoadSceneFromGlTF(filepath, *m_scene, *m_renderer);
		// TODO: include camera in the glTF
		m_scene->GetCamera().SetPosition(glm::vec3(0.0f, -6.0f, 3.0f));

		// Binding the descriptors to the new textures
		m_renderer->UpdateDescriptorSets();
		
		LOG("[Application] Scene loaded successfully!");
	}

	void Application::Update()
	{
		// Update user input
		m_input->Update(*m_window);

		// Camera control
		// TODO:
		// - fix the double sensitivity issue
		// - expose sensitivity in the UI
		if (!ImGui::GetIO().WantCaptureMouse)
		{
			auto& camera = m_scene->GetCamera();
			GLFWwindow* win = m_window->GetHandle();

			// Mouse buttons -> Panning and orbiting
			if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
				camera.Rotate(m_input->mouseDeltaX, m_input->mouseDeltaY);
			else if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
				camera.Pan(Input::PAN_SENSITIVITY * m_input->mouseDeltaX, Input::PAN_SENSITIVITY * m_input->mouseDeltaY);

			// Mouse scroll-wheel -> Dolly
			double mouseScroll = m_input->GetMouseScroll();
			if (mouseScroll != 0.0)
				camera.Dolly(mouseScroll);
		}

		// Update UI
		m_UI->Update(*m_scene, *this);
	}
}