#include <string>
#include <glm/glm.hpp>

namespace Felina
{
    struct Light
    {
        enum Type { DIRECTIONAL = 0, POINT, SPOT };

        std::string name;
        glm::vec3 color;
        float intensity;
        Type type;
        float range;

        // Spot only (they store the cosine of the angle)
        float innerConeAngle;
        float outerConeAngle;

        Light(Type type,
            const glm::vec3& color = { 1.0f, 1.0f, 1.0f },
            float intensity = 1.0f,
            float range = 0.0f, // 0.0 = infinite
            float inner = 0.0f,
            float outer = 1.0f,
            const std::string& name = "")
            :type(type), color(color), intensity(intensity), range(range),
            innerConeAngle(inner), outerConeAngle(outer), name(name)
        {
        }
    };
}