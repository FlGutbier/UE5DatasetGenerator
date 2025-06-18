// Copyright (c) 2025 Florian Gutbier
// 
// This source code is part of the UE5 Plugin developed for the Bachelor's thesis
// at the University of Bamberg.
// 
// Released under the MIT License. See LICENSE file for details.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "Engine/World.h"
#include "Engine/EngineTypes.h"
#include "GameFramework/Actor.h"
#include "GameFramework/PlayerController.h"
#include "Engine/Engine.h"
#include "Engine/TargetPoint.h"
#include "Engine/LocalFogVolume.h"
#include "Misc/Paths.h"
#include "HAL/PlatformFileManager.h"
#include "TimerManager.h"
#include "DatasetMetadataWriter.h"
#include "UDatasetCaptureManager.generated.h"

/**
 * @brief A manager class responsible for automating dataset generation.
 *
 * This class handles spawning actors, adjusting lighting, positioning the camera,
 * and taking screenshots from multiple perspectives under varying conditions.
 * It writes metadata for each image to a CSV file.
 */
UCLASS()
class DATASETRENDERERBACHELORTHESIS_API UDatasetCaptureManager : public UObject
{
	GENERATED_BODY()

public:

	/**
	 * @brief Initializes the capture manager with scene, actor, and camera configuration.
	 *
	 * @param InWorld The world context in which actors are spawned.
	 * @param InObjectTarget A reference point used for actor placement and camera orientation.
	 * @param InCameraTargets A list of directional vectors indicating camera positions.
	 * @param LightColors A list of light colors to apply during capture.
	 * @param InActorClassMap A map of actor types and associated class labels.
	 * @param NextLevel The level to load after capture is complete.
	 */
	void Initialize(
		UWorld* InWorld,
		ATargetPoint* InObjectTarget,
		const TArray<FVector>& InCameraTargets,
		const TArray<FLinearColor>& LightColors,
		const TArray<UMaterialInterface*>& Materials,
		const TMap<TSubclassOf<AActor>, int32>& InActorClassMap,
		const TSoftObjectPtr<UWorld>& NextLevel,
		bool addFog
	);

	void StartCapture();

private:

	// Store pointers as UPROPERTY to ensure proper functionality of garbage collector.

	UPROPERTY()
	UWorld* m_pWorld;

	UPROPERTY()
	ATargetPoint* m_pObjectTarget;

	UPROPERTY()
	TArray<FVector> m_aCameraTargets;

	UPROPERTY()
	TArray<UMaterialInterface*> m_aMaterials;

	UPROPERTY()
	TSoftObjectPtr<UWorld> m_pNextLevel;

	UPROPERTY()
	AActor* m_pCurrentSpawnedActor = nullptr;

	UPROPERTY()
	UDatasetMetadataWriter* m_pMetadataWriter;

	UPROPERTY()
	ALocalFogVolume* m_pLocalFog;

	TArray<TPair<TSubclassOf<AActor>, int32>> m_aActorEntries;

	TArray<FLinearColor> m_aLightColors;

	UPROPERTY()
	FVector m_vCurrentObjectCenter;

	float   m_fCurrentObjectRadius;

	int32 m_iCurrentActorIndex;

	int32 m_iCurrentCameraIndex;

	int32 m_iCurrentLightColorIndex;

	UPROPERTY()
	TArray<FVector> m_vCameraTransforms;

	int32 m_iCurrentMaterialIndex;

	bool m_bFogActive;

	bool m_bAddFog;

	FDelegateHandle m_oScreenshotCapturedHandle;

	FString m_sCurrentScreenshotPath;

	FString m_sRelativeImagePath;

	FString GetDatasetRoot() const;

	void SetupActorEntries(const TMap<TSubclassOf<AActor>, int32>& InActorClassMap);

	void SetupCameraTargets(const TArray<FVector>& InCameraTargets);

	void SetupLightColors(const TArray<FLinearColor>& LightColors);

	void CreateScreenshotFolder() const;

	void SetUpFog();

	void BuildCameraTargetsForCurrentActor();

	void ProcessCaptureState();

	void SpawnCurrentActor();

	void CaptureScreenshotForCurrentCamera();

	void RequestCameraScreenshot();

	ALocalFogVolume* CreateTempFogVolume(const FVector& Location, const FVector& UniformScale, float RadialDensity, float HeightDensity, const FLinearColor& Albedo, float PhaseG);

	void SetAllLightsColor(const FLinearColor& NewColor) const;

	void ApplyCurrentMaterial();

	void OnScreenshotCaptured(int32 Width, int32 Height, const TArray<FColor>& Bitmap);

	void FinalizeCapture();

	bool SavePNG(const FString& Filename, const TArray<FColor>& SrcBitmap, int32 Width, int32 Height);
};
