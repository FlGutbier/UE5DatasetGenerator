// Copyright (c) 2025 Florian Gutbier
// 
// This source code is part of the UE5 Plugin developed for the Bachelor's thesis
// at the University of Bamberg.
// 
// Released under the MIT License. See LICENSE file for details.

#include "DatasetRendererBachelorThesisBPLibrary.h"
#include "Engine/World.h"
#include "TimerManager.h"
#include "Engine/Engine.h"
#include "GameFramework/PlayerController.h"
#include "Misc/Paths.h"
#include "HAL/PlatformFilemanager.h"
#include "RenderingThread.h"
#include "UDatasetCaptureManager.h"


void UBachelorRenderingBPLibrary::StartDatasetCapture(
    UObject* WorldContextObject,
    ATargetPoint* ObjectTarget,
    const TArray<FVector>& CameraTargets,
    const TArray<FLinearColor>& LightColors,
    const TArray<UMaterialInterface*>& Materials,
    const TMap<TSubclassOf<AActor>, int32>& ActorClassMap,
    bool addFog,
    TSoftObjectPtr<UWorld> NextLevel
)
{
    if (!WorldContextObject)
    {
        UE_LOG(LogTemp, Warning, TEXT("StartDatasetCapture: WorldContextObject is null."));
        return;
    }

    UWorld* World = WorldContextObject->GetWorld();
    if (!World)
    {
        UE_LOG(LogTemp, Warning, TEXT("StartDatasetCapture: Could not get UWorld from context."));
        return;
    }

    // Create our manager as a transient UObject
    UDatasetCaptureManager* Manager = NewObject<UDatasetCaptureManager>();

    // Keep it from being garbage‐collected
    Manager->AddToRoot();

    // Initialize with user data
    Manager->Initialize(World, ObjectTarget, CameraTargets, LightColors, Materials, ActorClassMap, NextLevel, addFog);

    // Start the capture process
    Manager->StartCapture();

    // Done: the manager will handle iteration and eventually remove itself from root
}