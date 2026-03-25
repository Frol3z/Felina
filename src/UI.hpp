#pragma once

#include <glm/glm.hpp>
#include <filesystem>

namespace Felina
{
	class Object;
	class Scene;
	class Application;

	class UI
	{
		public:
			UI();
			void Update(Scene& scene, Application& app);

		private:
			void DrawSceneWindow(Scene& scene, Application& app);
			void DrawHierarchyObject(Object* object, size_t& idx);
			void DrawInspectorWindow();

			std::filesystem::path OpenFileDialog (const std::filesystem::path& defaultPath, const std::vector<const char *>& filters) const;

			// Temporary solution to center a button
			bool ButtonCenteredOnLine(const char* label, float alignment = 0.5f);

			Object* m_hierarchySelection = nullptr;
			glm::vec3 m_displayedPosition;
			glm::vec3 m_displayedRotation;
			glm::vec3 m_displayedScale;
	};
}