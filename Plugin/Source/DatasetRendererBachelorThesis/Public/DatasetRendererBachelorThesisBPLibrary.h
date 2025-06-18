// Copyright (c) 2025 Florian Gutbier
// 
// This source code is part of the UE5 Plugin developed for the Bachelor's thesis
// at the University of Bamberg.
// 
// Released under the MIT License. See LICENSE file for details.
#pragma once

#include "Kismet/BlueprintFunctionLibrary.h"
#include "Engine/World.h"
#include "Engine/EngineTypes.h"
#include "GameFramework/Actor.h"
#include "GameFramework/PlayerController.h"
#include "Engine/Engine.h"
#include "Misc/Paths.h"
#include "Engine/TargetPoint.h"
#include "HAL/PlatformFileManager.h"
#include "TimerManager.h"
#include "DatasetRendererBachelorThesisBPLibrary.generated.h"

/**
 * @brief Blueprint function library to trigger dataset rendering from Blueprints.
 *
 * Provides a single entry point for starting the dataset capture process
 * by instantiating and controlling the DatasetCaptureManager.
 */
UCLASS()
class UBachelorRenderingBPLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

public:

    /**
     * @brief Starts the dataset capture process.
     *
     * Spawns a DatasetCaptureManager instance, sets it up with the provided configuration,
     * and initiates actor, camera, lighting, and screenshot automation.
     *
     * @param WorldContextObject The world context (the object targetpoint can be used for this).
     * @param ObjectTarget A scene reference point for positioning the model.
     * @param CameraTargets An array of camera direction vectors.
     * @param LightColors An array of light colors to iterate over.
     * @param ActorClassMap A map of actor classes to their corresponding label/class ID.
     * @param NextLevel Optional. If provided, this level will be loaded after completion.
     */
    UFUNCTION(BlueprintCallable, Category = "Dataset", meta = (WorldContext = "WorldContextObject"))
    static void StartDatasetCapture(
        UObject* WorldContextObject,
        ATargetPoint* ObjectTarget,
        const TArray<FVector>& CameraTargets,
        const TArray<FLinearColor>& LightColors,
        const TArray<UMaterialInterface*>& Materials,
        const TMap<TSubclassOf<AActor>, int32>& ActorClassMap,
        bool addFog,
        TSoftObjectPtr<UWorld> NextLevel = nullptr
    );
};
