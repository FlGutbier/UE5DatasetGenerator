// Copyright (c) 2025 Florian Gutbier
// 
// This source code is part of the UE5 Plugin developed for the Bachelor's thesis
// at the University of Bamberg.
// 
// Released under the MIT License. See LICENSE file for details.

#pragma once

#include "CoreMinimal.h"
#include "DatasetMetadataWriter.generated.h"

/**
 * @brief A utility class responsible for writing metadata related to dataset rendering to a file.
 *
 * This includes metadata such as the model name, level, material, lighting, and camera information.
 */
UCLASS()
class DATASETRENDERERBACHELORTHESIS_API UDatasetMetadataWriter : public UObject
{
	GENERATED_BODY()

public:

	/**
	 * @brief Initializes metadata fields to default values.
	 */
	void Initialize();

	/**
	 * @brief Sets the name of the model.
	 * @param modelName The name of the model.
	 */
	inline void setModelName(FString modelName) {
		m_sModelName = modelName;
	}

	/**
	 * @brief Sets the name of the level.
	 * @param levelName The name of the level.
	 */
	inline void setLevelName(FString levelName) {
		m_sLevelName = levelName;
	}

	/**
	 * @brief Sets the name of the material.
	 * @param materialName The name of the material.
	 */
	inline void setMaterialName(FString materialName) {
		m_sMaterialName = materialName;
	}

	/**
	 * @brief Enables or disables fog in the metadata.
	 * @param isFogEnabled True if fog is enabled, false otherwise.
	 */
	inline void setIsFogEnabled(bool isFogEnabled) {
		m_sIsFogEnabled = isFogEnabled ? "true" : "false";
	}

	/**
	 * @brief Sets the position of the camera.
	 * @param cameraPosition A string representing the camera's position.
	 */
	inline void setCameraPosition(FString cameraPosition) {
		m_sCameraPosition = cameraPosition;
	}

	/**
	 * @brief Sets the name of the image.
	 * @param imagePath The file name or path of the image.
	 */
	inline void setImageName(FString imagePath) {
		m_sImageName = imagePath;
	}

	/**
	 * @brief Sets the path of the metadata output file.
	 * @param filePath The path where the file will be written.
	 */
	inline void setFilePath(FString filePath) {
		m_sFilePath = filePath;
	}

	/**
	 * @brief Sets the color of the light in the scene.
	 * @param lightColor A string representation of the light color.
	 */
	inline void setLightColor(FString lightColor)
	{
		m_sLightColor = lightColor;
	}

	inline void setClassIndex(FString classIndex) {
		m_sClassIndex = classIndex;
	}

	/**
	 * @brief Creates a metadata file and writes the header if it does not exist.
	 */
	void CreateFile() const;

	/**
	 * @brief Appends a line of metadata to the file.
	 */
	void WriteToFile() const;

private:
	/** The name of the 3D model */
	FString m_sModelName;

	/** The name of the level/map */
	FString m_sLevelName;

	/** The name of the material used */
	FString m_sMaterialName;

	/** The color of the lighting */
	FString m_sLightColor;

	/** The position of the camera */
	FString m_sCameraPosition;

	/** The name or path of the rendered image */
	FString m_sImageName;

	/** The file path where metadata will be saved */
	FString m_sFilePath;

	/** Whether fog is enabled in the scene */
	FString m_sIsFogEnabled;

	FString m_sClassIndex;

};
