#include "DatasetRendererBachelorThesisBPLibrary.h"
#include "Engine/World.h"
#include "TimerManager.h"
#include "Engine/Engine.h"
#include "GameFramework/PlayerController.h"
#include "Misc/Paths.h"
#include "HAL/PlatformFilemanager.h"
#include "RenderingThread.h"
#include "UDatasetCaptureManager.h"


// We'll keep a single static delegate handle for the screenshot callback
void UBachelorRenderingBPLibrary::StartDatasetCapture(
    UObject* WorldContextObject,
    ATargetPoint* ObjectTarget,
    const TArray<FVector>& CameraTargets,
    const TArray<FLinearColor>& LightColors,
    const TMap<TSubclassOf<AActor>, int32>& ActorClassMap,
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
    Manager->Initialize(World, ObjectTarget, CameraTargets, LightColors, ActorClassMap, NextLevel);

    // Start the capture process
    Manager->StartCapture();

    // Done: the manager will handle iteration and eventually remove itself from root
}