#include "UI.hpp"

#include "Application.hpp"
#include "Scene.hpp"
#include "Common.hpp"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <tinyfiledialogs.h>

namespace Felina
{
	UI::UI()
		: m_displayedPosition(glm::vec3(0.0f)), m_displayedRotation(glm::vec3(0.0f)), m_displayedScale(glm::vec3(0.0f))
	{}

	void UI::Update(Scene& scene, Application& app)
	{
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

		//ImGui::ShowDemoWindow();

		// Custom UI
		DrawSceneWindow(scene, app);
		DrawInspectorWindow();

		ImGui::Render();
	}

	void UI::DrawSceneWindow(Scene& scene, Application& app)
	{
		ImGui::Begin("Scene");

		// Load new scene button
		if (ButtonCenteredOnLine("Load New Scene...", 0.5f))
		{
			auto selectedFilePath = OpenFileDialog(ASSETS_DIR, { "*.glb", "*.gltf" });
			if (!selectedFilePath.empty())
			{
				app.LoadScene(selectedFilePath);
				m_hierarchySelection = nullptr; // This pointer may point to an old object
			}
		}

		// Draw scene hierarchy
		ImGui::SeparatorText("Hierarchy");
		size_t idx = 0; // Same reference trick used in Renderer.cpp
		for (const auto& obj : scene.GetObjects())
		{
			DrawHierarchyObject(obj.get(), idx);
		}

		// Camera stuff
		ImGui::SeparatorText("Camera");
		
		Camera& camera = scene.GetCamera();
		glm::vec3 pos = camera.GetPosition();
		glm::vec3 target = camera.GetTarget();
		float exposure = camera.GetExposure() * 100.0; // rescaling
		float skyboxIntensity = camera.GetSkyboxIntensity();

		ImGui::TextDisabled("Camera: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);
		ImGui::TextDisabled("Target: (%.2f, %.2f, %.2f)", target.x, target.y, target.z);
		if (ImGui::DragFloat("Exposure", &exposure, 0.001f, 0.0f, 1.0f))
			camera.SetExposure(exposure * 0.01); // rescaling
		if (ImGui::DragFloat("Skybox Intensity", &skyboxIntensity, 0.01f, 0.0f, 1.0f))
			camera.SetSkyboxIntensity(skyboxIntensity);
		
		ImGui::End();
	}

	void UI::DrawHierarchyObject(Object* obj, size_t& idx)
	{
		bool isLeaf = obj->GetChildren().empty();
		ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow
			| ImGuiTreeNodeFlags_OpenOnDoubleClick
			| ImGuiTreeNodeFlags_SpanAvailWidth
			| ImGuiTreeNodeFlags_DrawLinesFull
			| ImGuiTreeNodeFlags_DefaultOpen
			| (isLeaf ? ImGuiTreeNodeFlags_Leaf : 0)
			| ((m_hierarchySelection == obj) ? ImGuiTreeNodeFlags_Selected : 0);
		
		bool isNodeOpen = ImGui::TreeNodeEx((void*)(intptr_t)idx, flags, obj->GetName().c_str());
		
		// Handle selection
		if (ImGui::IsItemClicked())
		{
			m_hierarchySelection = obj;

			// Initialize inspector values
			m_displayedPosition = obj->GetPosition();
			m_displayedRotation = glm::degrees(glm::eulerAngles(obj->GetRotation()));
			m_displayedScale = obj->GetScale();
		}
		
		// Increment the index AFTER drawing the object
		++idx;

		// If the node is open iterate recursively through its children
		if (isNodeOpen)
		{
			for (const auto& child : obj->GetChildren())
				DrawHierarchyObject(child.get(), idx);
			ImGui::TreePop();
		}
	}

	void UI::DrawInspectorWindow()
	{
		ImGui::Begin("Inspector");
		if (m_hierarchySelection)
		{
			ImGui::SeparatorText(m_hierarchySelection->GetName().c_str());

			// Transform
			// TODO: highlight in the UI that this is the transform-dedicated section
			if (ImGui::DragFloat3("Position", &m_displayedPosition.x, 0.01f))
			{
				m_hierarchySelection->SetPosition(m_displayedPosition);
			}

			if (ImGui::DragFloat3("Rotation", &m_displayedRotation.x, 1.0f))
			{
				glm::quat targetRotation = glm::quat(glm::radians(m_displayedRotation));
				m_hierarchySelection->SetRotation(targetRotation);
			}

			if (ImGui::DragFloat3("Scale", &m_displayedScale.x, 0.01f))
			{
				m_hierarchySelection->SetScale(m_displayedScale);
			}

			// TODO: this code should be readapted to maybe listing all the loaded resources
			// Mesh
			auto& rm = ResourceManager::GetInstance();
			const std::vector<MeshID>& meshes = m_hierarchySelection->GetMeshes();

			if(meshes.size())
			{
				if (ImGui::BeginCombo("Mesh", rm.GetMeshName(meshes[0]).c_str(), 0))
				{
					static ImGuiTextFilter filter;
					if (ImGui::IsWindowAppearing())
					{
						ImGui::SetKeyboardFocusHere();
						filter.Clear();
					}
					ImGui::SetNextItemShortcut(ImGuiMod_Ctrl | ImGuiKey_F);
					filter.Draw("##Filter", -FLT_MIN);

					for (MeshID meshId : meshes)
					{
						if (meshId == meshes[0])
							continue;

						auto& name = rm.GetMeshName(meshId);
						if (filter.PassFilter(name.c_str()))
							ImGui::Selectable(name.c_str(), false);
					}
					ImGui::EndCombo();
				}
				
				/*
				// Material
				MaterialID selectedMaterial = m_hierarchySelection->GetMaterial();
				if (ImGui::BeginCombo("Material", rm.GetMaterialName(selectedMaterial).c_str(), 0))
				{
					static ImGuiTextFilter filter;
					if (ImGui::IsWindowAppearing())
					{
						ImGui::SetKeyboardFocusHere();
						filter.Clear();
					}
					ImGui::SetNextItemShortcut(ImGuiMod_Ctrl | ImGuiKey_F);
					filter.Draw("##Filter", -FLT_MIN);

					for (auto& [id, material] : rm.GetMaterials())
					{
						auto& name = rm.GetMaterialName(id);
						const bool isSelected = (id == selectedMaterial);
						if (filter.PassFilter(name.c_str()))
							if (ImGui::Selectable(name.c_str(), isSelected))
								m_hierarchySelection->SetMaterial(id);
					}
					ImGui::EndCombo();
				}
				*/
			}
		}
		ImGui::End();
	}

	std::filesystem::path UI::OpenFileDialog(const std::filesystem::path& defaultPath, const std::vector<const char *>& filters) const
	{
		const char* selectedPath = tinyfd_openFileDialog(
			"",														// title
			std::filesystem::absolute(ASSETS_DIR).string().c_str(),	// default path or file (std::absolute makes subfolder paths work)
			filters.size(),											// filters count
			filters.data(),											// filters data
			"",														// description
			0														// allow multiple selections
		);

		// Check if the dialog has been closed without selecting anything
		if (selectedPath == nullptr)
			return {};
		return selectedPath;
	}

	// alignment: 0.0f -> align left
	// alignment: 0.5f -> align center
	// alignment: 1.0f -> align right
	bool UI::ButtonCenteredOnLine(const char* label, float alignment)
	{
		ImGuiStyle& style = ImGui::GetStyle();

		float size = ImGui::CalcTextSize(label).x + style.FramePadding.x * 2.0f;
		float avail = ImGui::GetContentRegionAvail().x;

		float off = (avail - size) * alignment;
		if (off > 0.0f)
			ImGui::SetCursorPosX(ImGui::GetCursorPosX() + off);

		return ImGui::Button(label);
	}
}