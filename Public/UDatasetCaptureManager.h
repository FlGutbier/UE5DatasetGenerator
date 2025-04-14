#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "Engine/World.h"
#include "Engine/EngineTypes.h"
#include "GameFramework/Actor.h"
#include "GameFramework/PlayerController.h"
#include "Engine/Engine.h"
#include "Engine/TargetPoint.h"
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
		const TMap<TSubclassOf<AActor>, int32>& InActorClassMap,
		const TSoftObjectPtr<UWorld>& NextLevel
	);

	/**
	 * @brief Begins the capture sequence.
	 */
	void StartCapture();

private:

	// Store pointers as UPROPERTY to ensure proper functionality of garbage collector.

	/** Pointer to the current world */
	UPROPERTY()
	UWorld* m_pWorld;

	/** The object placement target point */
	UPROPERTY()
	ATargetPoint* m_pObjectTarget;

	/** List of calculated camera positions in world space */
	UPROPERTY()
	TArray<FVector> m_aCameraTargets;

	/** Pointer to the level that should be loaded next */
	UPROPERTY()
	TSoftObjectPtr<UWorld> m_pNextLevel;

	/** The currently spawned actor during capture */
	UPROPERTY()
	AActor* m_pCurrentSpawnedActor = nullptr;

	/** Metadata writer used to generate CSV output */
	UPROPERTY()
	UDatasetMetadataWriter* m_pMetadataWriter;

	/** List of actor types and class labels for dataset labeling */
	TArray<TPair<TSubclassOf<AActor>, int32>> m_aActorEntries;

	/** List of light colors to cycle through */
	TArray<FLinearColor> m_aLightColors;

	/** Current actor index being processed */
	int32 m_iCurrentActorIndex = 0;

	/** Current camera index being processed */
	int32 m_iCurrentCameraIndex = 0;

	/** Current light color index being applied */
	int32 m_iCurrentLightColorIndex;

	/** Screenshot delegate handle */
	FDelegateHandle m_oScreenshotCapturedHandle;

	/** Path to the current screenshot file being generated */
	FString m_sCurrentScreenshotPath;

	/** Converts the input map to a linear list of actor-class pairs */
	void SetupActorEntries(const TMap<TSubclassOf<AActor>, int32>& InActorClassMap);

	/** Computes camera positions relative to the object target */
	void SetupCameraTargets(const TArray<FVector>& InCameraTargets);

	/** Stores the list of lighting conditions to apply */
	void SetupLightColors(const TArray<FLinearColor>& LightColors);

	/** Creates the folder for storing screenshots and metadata */
	void CreateScreenshotFolder() const;

	/** Core state machine controlling capture flow */
	void ProcessCaptureState();

	/** Spawns the actor for the current iteration */
	void SpawnCurrentActor();

	/** Initiates a screenshot for the current camera position */
	void CaptureScreenshotForCurrentCamera();

	/** Requests the engine to take a screenshot */
	void RequestCameraScreenshot();

	/** Sets all dynamic light components in the scene to the given color */
	void SetAllLightsColor(const FLinearColor& NewColor) const;

	/** Called when a screenshot has been taken */
	void OnScreenshotCaptured(int32 Width, int32 Height, const TArray<FColor>& Bitmap);

	/** Finalizes the process and optionally loads the next level */
	void FinalizeCapture();
};
