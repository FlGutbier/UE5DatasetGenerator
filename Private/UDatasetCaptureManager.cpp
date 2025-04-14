// Fill out your copyright notice in the Description page of Project Settings.


#include "UDatasetCaptureManager.h"
#include "Kismet/GameplayStatics.h"
#include "Components/LightComponent.h"
#include "Kismet/KismetMathLibrary.h"
#include "Engine/Light.h"

/***********************************************************************************************/
void UDatasetCaptureManager::Initialize(
    UWorld* InWorld,
    ATargetPoint* InObjectTarget,
    const TArray<FVector>& InCameraTargets,
    const TArray<FLinearColor>& LightColors,
    const TMap<TSubclassOf<AActor>, int32>& InActorClassMap,
    const TSoftObjectPtr<UWorld>& NextLevel)
{
    m_pWorld = InWorld;
    m_pObjectTarget = InObjectTarget;
    m_pNextLevel = NextLevel;

    // Must be initialized before creating screenshot folder
    m_pMetadataWriter = NewObject<UDatasetMetadataWriter>();
    m_pMetadataWriter->Initialize();

    m_sCurrentScreenshotPath = "";
    m_iCurrentActorIndex = 0;
    m_iCurrentCameraIndex = 0;
    m_iCurrentLightColorIndex = 0;
    m_pCurrentSpawnedActor = nullptr;

    SetupActorEntries(InActorClassMap);
    SetupCameraTargets(InCameraTargets);
    SetupLightColors(LightColors);
    CreateScreenshotFolder();
}

/***********************************************************************************************/
void UDatasetCaptureManager::SetupActorEntries(const TMap<TSubclassOf<AActor>, int32>& InActorClassMap)
{
    m_aActorEntries.Reset();
    for (const TPair<TSubclassOf<AActor>, int32>& Pair : InActorClassMap)
    {
        m_aActorEntries.Add(Pair);
    }
    m_iCurrentActorIndex = 0;
}

/***********************************************************************************************/
void UDatasetCaptureManager::SetupCameraTargets(const TArray<FVector>& InCameraTargets)
{
    // Reset camera targets and calculate new positions based on object target
    m_aCameraTargets.Reset();
    if (m_pObjectTarget)
    {
        FTransform SpawnTransform = m_pObjectTarget->GetActorTransform();
        for (const FVector& Dir : InCameraTargets)
        {
            FVector NormDir = Dir.GetSafeNormal();
            FVector WorldDir = SpawnTransform.GetRotation().RotateVector(NormDir);
            FVector Offset = WorldDir * 150.f; // 150 cm offset
            FVector CameraLoc = SpawnTransform.GetLocation() + Offset;
            m_aCameraTargets.Add(CameraLoc);
        }
    }
    m_iCurrentCameraIndex = 0;
}

/***********************************************************************************************/
void UDatasetCaptureManager::SetupLightColors(const TArray<FLinearColor>& LightColors)
{
    m_aLightColors = LightColors;
    m_iCurrentLightColorIndex = 0;
}

/***********************************************************************************************/
void UDatasetCaptureManager::CreateScreenshotFolder() const
{
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    FString FolderPath = FPaths::ProjectDir() / TEXT("Dataset");

    if (!PlatformFile.DirectoryExists(*FolderPath))
    {
        PlatformFile.CreateDirectoryTree(*FolderPath);
    }

    FString CSVPath = FPaths::Combine(FolderPath, TEXT("Metadata.csv"));
    m_pMetadataWriter->setFilePath(CSVPath);
}

/***********************************************************************************************/
void UDatasetCaptureManager::StartCapture()
{
    m_pMetadataWriter->CreateFile();
    m_pMetadataWriter->setLevelName(m_pWorld->GetMapName());
    ProcessCaptureState();
}

/***********************************************************************************************/
void UDatasetCaptureManager::ProcessCaptureState()
{
    // Check that the world is valid.
    if (!m_pWorld)
    {
        UE_LOG(LogTemp, Warning, TEXT("Capture Manager: World is invalid."));
        // Proper cleanup if needed.
        FinalizeCapture();
        return;
    }

    // If all actors have been processed, finalize capture.
    if (m_iCurrentActorIndex >= m_aActorEntries.Num())
    {
        FinalizeCapture();
        return;
    }

    // If there is no currently spawned actor for this actor index, spawn it.
    if (!m_pCurrentSpawnedActor)
    {
        SpawnCurrentActor();
        return;
    }

    // Process camera iteration for the current actor
    if (m_iCurrentCameraIndex >= m_aCameraTargets.Num())
    {
        // All cameras for this light color have been processed; go to the next light color.
        m_iCurrentLightColorIndex++;
        if (m_aLightColors.IsValidIndex(m_iCurrentLightColorIndex))
        {
            SetAllLightsColor(m_aLightColors[m_iCurrentLightColorIndex]);
            m_iCurrentCameraIndex = 0; // reset camera index
        }
        else
        {
            // All iterations completed for current actor: destroy actor and move on.
            if (!m_pCurrentSpawnedActor->IsPendingKillPending())
            {
                m_pCurrentSpawnedActor->Destroy();
            }
            m_pCurrentSpawnedActor = nullptr;
            m_iCurrentActorIndex++;
            // Reset indices for next actor.
            m_iCurrentLightColorIndex = 0;
            m_iCurrentCameraIndex = 0;
            ProcessCaptureState();
            return;
        }
    }

    // Process a camera capture.
    CaptureScreenshotForCurrentCamera();
}

/***********************************************************************************************/
void UDatasetCaptureManager::SpawnCurrentActor()
{
    const TPair<TSubclassOf<AActor>, int32>& CurrentEntry = m_aActorEntries[m_iCurrentActorIndex];
    TSubclassOf<AActor> ActorClass = CurrentEntry.Key;
    if (!ActorClass)
    {
        UE_LOG(LogTemp, Warning, TEXT("Capture Manager: Invalid ActorClass at index %d"), m_iCurrentActorIndex);
        m_iCurrentActorIndex++;
        ProcessCaptureState();
        return;
    }

    FTransform SpawnTransform = m_pObjectTarget ? m_pObjectTarget->GetActorTransform() : FTransform::Identity;
    m_pCurrentSpawnedActor = m_pWorld->SpawnActor<AActor>(ActorClass, SpawnTransform);
    if (!m_pCurrentSpawnedActor)
    {
        UE_LOG(LogTemp, Warning, TEXT("Capture Manager: Failed to spawn actor of class %s"), *ActorClass->GetName());
        m_iCurrentActorIndex++;
        ProcessCaptureState();
        return;
    }

    UE_LOG(LogTemp, Log, TEXT("Capture Manager: Spawned %s"), *m_pCurrentSpawnedActor->GetName());
    m_pMetadataWriter->setModelName(ActorClass->GetName());

    // Reset indices for the new actor.
    m_iCurrentCameraIndex = 0;
    m_iCurrentLightColorIndex = 0;
    if (m_aLightColors.IsValidIndex(m_iCurrentLightColorIndex))
    {
        SetAllLightsColor(m_aLightColors[m_iCurrentLightColorIndex]);
    }
    ProcessCaptureState(); // Proceed with capturing after the actor is spawned.
}

/***********************************************************************************************/
void UDatasetCaptureManager::CaptureScreenshotForCurrentCamera()
{
    const FVector& CamLocation = m_aCameraTargets[m_iCurrentCameraIndex];
    APlayerController* PC = m_pWorld->GetFirstPlayerController();
    if (PC)
    {
        
        if (APawn* Pawn = PC->GetPawn())
        {
            FVector TargetLocation = m_pCurrentSpawnedActor->GetActorLocation();
            FRotator LookAtRot = UKismetMathLibrary::FindLookAtRotation(CamLocation, TargetLocation);
            Pawn->SetActorLocation(CamLocation);
            PC->SetControlRotation(LookAtRot);
        }
    }

    m_pMetadataWriter->setCameraPosition(CamLocation.ToString());

    // Prepare the per-class subfolder and final filename.
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    const int32 ClassIndex = m_aActorEntries[m_iCurrentActorIndex].Value;
    FString ClassFolder = FPaths::Combine(FPaths::ProjectDir(), TEXT("Dataset"), FString::FromInt(ClassIndex), m_pWorld->GetMapName());
    PlatformFile.CreateDirectoryTree(*ClassFolder);

    FString Filename = FString::Printf(
        TEXT("%s/%s_%d_%d_%s.bmp"),
        *ClassFolder,
        *m_pCurrentSpawnedActor->GetName(),
        m_iCurrentCameraIndex,
        m_iCurrentLightColorIndex,
        *m_pWorld->GetMapName()
    );
    m_sCurrentScreenshotPath = Filename;

    // Schedule the screenshot request after a short delay.
    FTimerHandle DelayHandle;
    m_pWorld->GetTimerManager().SetTimer(DelayHandle, this, &UDatasetCaptureManager::RequestCameraScreenshot, 0.1f, false);
}

/***********************************************************************************************/
void UDatasetCaptureManager::RequestCameraScreenshot()
{
    if (UGameViewportClient* ViewportClient = m_pWorld->GetGameViewport())
    {
        // Remove any old handle just in case.
        if (m_oScreenshotCapturedHandle.IsValid())
        {
            ViewportClient->OnScreenshotCaptured().Remove(m_oScreenshotCapturedHandle);
            m_oScreenshotCapturedHandle.Reset();
        }
        m_oScreenshotCapturedHandle = ViewportClient->OnScreenshotCaptured().AddUObject(
            this,
            &UDatasetCaptureManager::OnScreenshotCaptured
        );
        FScreenshotRequest::RequestScreenshot(m_sCurrentScreenshotPath, false, false);
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("Capture Manager: No GameViewportClient found."));
        m_iCurrentCameraIndex++;
        ProcessCaptureState();
    }
}

/***********************************************************************************************/
void UDatasetCaptureManager::SetAllLightsColor(const FLinearColor& NewColor) const
{
    if (!m_pWorld) return;

    // Get all ALight actors in the level
    TArray<AActor*> FoundActors;
    UGameplayStatics::GetAllActorsOfClass(m_pWorld, ALight::StaticClass(), FoundActors);

    for (AActor* Actor : FoundActors)
    {
        if (const ALight* Light = Cast<ALight>(Actor))
        {
            // Get the Light Component and set the new color
            if (ULightComponent* LightComponent = Light->GetLightComponent())
            {
                LightComponent->SetLightColor(NewColor);
            }
        }
    }
    m_pMetadataWriter->setLightColor(NewColor.ToString());
}

/***********************************************************************************************/
void UDatasetCaptureManager::OnScreenshotCaptured(int32 Width, int32 Height, const TArray<FColor>& Bitmap)
{
    if (m_pWorld)
    {
        if (UGameViewportClient* ViewportClient = m_pWorld->GetGameViewport())
        {
            ViewportClient->OnScreenshotCaptured().Remove(m_oScreenshotCapturedHandle);
        }
    }
    m_oScreenshotCapturedHandle.Reset();

    if (!m_sCurrentScreenshotPath.IsEmpty() && Bitmap.Num() == Width * Height)
    {
        bool bSaved = FFileHelper::CreateBitmap(*m_sCurrentScreenshotPath, Width, Height, (FColor*)Bitmap.GetData());
        UE_LOG(LogTemp, Log, TEXT("Capture Manager: Screenshot saved: %s => %s"),
            *m_sCurrentScreenshotPath, bSaved ? TEXT("SUCCESS") : TEXT("FAIL"));
        if (bSaved)
        {
            FString ImageName = FPaths::GetCleanFilename(m_sCurrentScreenshotPath);
            m_pMetadataWriter->setImageName(ImageName);
            m_pMetadataWriter->WriteToFile();
        }
    }

    // Move on: increment camera index and trigger the next state.
    m_iCurrentCameraIndex++;
    ProcessCaptureState();
}

/***********************************************************************************************/
void UDatasetCaptureManager::FinalizeCapture()
{
    if (m_pWorld)
    {
        UE_LOG(LogTemp, Log, TEXT("Capture Manager: All Actors processed!"));
    }
    else
    {
        UE_LOG(LogTemp, Log, TEXT("Capture Manager: world not valid!"));
    }
    

    if (!m_pNextLevel.IsNull())
    {
        FSoftObjectPath SoftObjectPath = m_pNextLevel.ToSoftObjectPath();
        FName LevelToLoad(*SoftObjectPath.GetLongPackageName());
        if (!LevelToLoad.IsNone())
        {
            UE_LOG(LogTemp, Log, TEXT("Capture Manager: Opening next level: %s"), *LevelToLoad.ToString());
            UGameplayStatics::OpenLevel(m_pWorld, LevelToLoad);
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("Capture Manager: NextLevel was valid but returned None package name?"));
        }
    }
    else
    {
        UE_LOG(LogTemp, Log, TEXT("Capture Manager: No next level provided (null), skipping level load."));
    }

    RemoveFromRoot();
}