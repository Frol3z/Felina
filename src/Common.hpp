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
	constexpr uint8_t MAX_LIGHTS = 16;
	constexpr uint16_t CUBEMAP_RESOLUTION = 1024;

	const std::filesystem::path DEFAULT_SCENE{ "./assets/conference.glb" };
	const std::filesystem::path SKYBOX{ "./assets/skybox/qwantani_dusk_2_puresky_4k.hdr" };
	const std::filesystem::path ASSETS_DIR{ "./assets/" };

	// NOTE: originally designed to read SPIR-V file, so it
	// may need adjustments reading other file formats is required
	std::vector<char> ReadFile(const std::string& filepath);
}