#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <filesystem>

#define LOG(x) std::cout << x << '\n'

namespace Felina
{
	using MeshID = uint32_t;
	using MaterialID = uint32_t;
	using TextureID = uint32_t;

	// NOTE: for a greater number of concurrent frames
	// the CPU might get ahead of the GPU causing latency
	// between frames
	constexpr uint8_t MAX_FRAMES_IN_FLIGHT = 2;
	constexpr uint8_t MAX_OBJECTS = 200; // max number of drawable objects
	constexpr uint8_t MAX_MATERIALS = 128;
	constexpr uint8_t MAX_TEXTURES = 136; // skybox not included

	const std::filesystem::path DEFAULT_SCENE{ "./assets/complex_hierarchy.glb" };
	const std::filesystem::path SKYBOX_DIR{ "./assets/skybox/" };
	const std::filesystem::path ASSETS_DIR{ "./assets/" };

	// NOTE: originally designed to read SPIR-V file, so it
	// may need adjustments reading other file formats is required
	std::vector<char> ReadFile(const std::string& filepath);
}