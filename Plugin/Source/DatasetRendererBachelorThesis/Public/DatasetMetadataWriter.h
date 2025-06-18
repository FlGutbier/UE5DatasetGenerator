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
	void Initialize();

	inline void setModelName(FString modelName) {
		m_sModelName = modelName;
	}

	inline void setLevelName(FString levelName) {
		m_sLevelName = levelName;
	}

	inline void setMaterialName(FString materialName) {
		m_sMaterialName = materialName;
	}

	inline void setIsFogEnabled(bool isFogEnabled) {
		m_sIsFogEnabled = isFogEnabled ? "true" : "false";
	}

	inline void setCameraPosition(FString cameraPosition) {
		m_sCameraPosition = cameraPosition;
	}

	inline void setImageName(FString imagePath) {
		m_sImageName = imagePath;
	}

	inline void setFilePath(FString filePath) {
		m_sFilePath = filePath;
	}

	inline void setLightColor(FString lightColor)
	{
		m_sLightColor = lightColor;
	}

	inline void setClassIndex(FString classIndex) {
		m_sClassIndex = classIndex;
	}

	void CreateFile() const;

	void WriteToFile() const;

private:

	FString m_sModelName;

	FString m_sLevelName;

	FString m_sMaterialName;

	FString m_sLightColor;

	FString m_sCameraPosition;

	FString m_sImageName;

	FString m_sFilePath;

	FString m_sIsFogEnabled;

	FString m_sClassIndex;

};
