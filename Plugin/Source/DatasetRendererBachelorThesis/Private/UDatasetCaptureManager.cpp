// Copyright (c) 2025 Florian Gutbier
// 
// This source code is part of the UE5 Plugin developed for the Bachelor's thesis
// at the University of Bamberg.
// 
// Released under the MIT License. See LICENSE file for details.


#include "UDatasetCaptureManager.h"
#include "Kismet/GameplayStatics.h"
#include "IImageWrapperModule.h"
#include "IImageWrapper.h"
#include "Components/LightComponent.h"
#include "Components/LocalFogVolumeComponent.h"
#include "Engine/LocalFogVolume.h"
#include "Kismet/KismetMathLibrary.h"
#include "Engine/Light.h"

/***********************************************************************************************/
void UDatasetCaptureManager::Initialize(
    UWorld* World,
    ATargetPoint* ObjectTarget,
    const TArray<FVector>& CameraTargets,
    const TArray<FLinearColor>& LightColors,
    const TArray<UMaterialInterface*>& Materials,
    const TMap<TSubclassOf<AActor>, int32>& InActorClassMap,
    const TSoftObjectPtr<UWorld>& NextLevel,
    bool bAddFog)
{
    if (!ensureAlwaysMsgf(World && ObjectTarget,
        TEXT("DatasetCaptureManager::Initialize called with null world or target")))
    {
        return;
    }

    // Initialize members
    m_pWorld = World;
    m_pObjectTarget = ObjectTarget;
    m_pNextLevel = NextLevel;
    m_vCameraTransforms = CameraTargets;
    m_aMaterials = Materials;
    m_aLightColors = LightColors;
    m_bAddFog = bAddFog;
    
    m_sCurrentScreenshotPath = "";
    m_sRelativeImagePath = "";
    m_iCurrentActorIndex = 0;
    m_iCurrentCameraIndex = 0;
    m_iCurrentLightColorIndex = 0;
    m_iCurrentMaterialIndex = 0;
    m_bFogActive = false;
    m_fCurrentObjectRadius = 0.f;
    m_vCurrentObjectCenter = FVector::ZeroVector;
    m_pCurrentSpawnedActor = nullptr;

    // Metadatawriter must be initialized before screenshot folder
    m_pMetadataWriter = NewObject<UDatasetMetadataWriter>();
    check(m_pMetadataWriter);
    m_pMetadataWriter->Initialize();

    SetupActorEntries(InActorClassMap);
    // SetupCameraTargets(InCameraTargets);  // Uncomment if using constant distance with camera positions (also remove calls to BuildCameraTargetsForCurrentActor()).
    CreateScreenshotFolder();

    if (m_bAddFog)
    {
        SetUpFog();
    }
}

FString UDatasetCaptureManager::GetDatasetRoot() const
{
    return FPaths::ProjectDir() / TEXT("Dataset");
}

/***********************************************************************************************/
void UDatasetCaptureManager::SetupActorEntries(
    const TMap<TSubclassOf<AActor>, int32>& InActorClassMap)
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
    m_aCameraTargets.Reset();
    if (!m_pObjectTarget) return;

    constexpr float  CameraDistance = 150.f; // cm
    const FTransform ObjectXform = m_pObjectTarget->GetActorTransform();

    for (const FVector& Dir : InCameraTargets)
    {
        const FVector WorldOffset =
            ObjectXform.TransformVectorNoScale(Dir.GetSafeNormal()) * CameraDistance;

        m_aCameraTargets.Add(ObjectXform.GetLocation() + WorldOffset);
    }
    m_iCurrentCameraIndex = 0;
}

/*----------------------------------------------------------------------------*/
void UDatasetCaptureManager::CreateScreenshotFolder() const
{
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    const FString  DatasetRoot = GetDatasetRoot();

    if (!PlatformFile.DirectoryExists(*DatasetRoot))
    {
        PlatformFile.CreateDirectoryTree(*DatasetRoot);
    }

    m_pMetadataWriter->setFilePath(FPaths::Combine(DatasetRoot, TEXT("Metadata.csv")));
}

/***********************************************************************************************/
void UDatasetCaptureManager::SetUpFog()
{
    if (!m_bAddFog || !m_pWorld || !m_pObjectTarget) return;

    // layout-related constants
    constexpr float  FogZOffset = 100.f;                    // +1 m up
    constexpr float  SphereRadius = 400.f;                    // 8 m diameter
    const FVector    SpawnLoc = m_pObjectTarget->GetActorLocation() +
        FVector(0.f, 0.f, FogZOffset);

    // visual look
    constexpr float        RadialDensity = 2.0f;
    constexpr float        HeightDensity = 2.0f;
    constexpr float        PhaseG = 0.15f;               // forward scatter
    const FLinearColor     Albedo = { 0.75f, 0.75f, 0.85f }; 

    m_pLocalFog = CreateTempFogVolume(
        SpawnLoc,
        FVector(SphereRadius),
        RadialDensity,
        HeightDensity,
        Albedo,
        PhaseG);

    if (m_pLocalFog)
    {
        m_pLocalFog->SetActorHiddenInGame(true);   //stay hidden until needed
    }
}

/***********************************************************************************************/
void UDatasetCaptureManager::StartCapture()
{
    if (!ensureMsgf(m_pMetadataWriter, TEXT("MetadataWriter is null"))) return;

    m_pMetadataWriter->CreateFile();
    m_pMetadataWriter->setLevelName(m_pWorld ? m_pWorld->GetMapName() : TEXT("Unknown"));

    ProcessCaptureState(); // process capture states
}

// ============================================================================
// Capture state machine
// ============================================================================
void UDatasetCaptureManager::ProcessCaptureState()
{
    // ---- Guard clauses ----
    if (!m_pWorld)
    {
        UE_LOG(LogTemp, Warning, TEXT("Capture Manager: World is invalid."));
        FinalizeCapture();
        return;
    }

    if (m_iCurrentActorIndex >= m_aActorEntries.Num())
    {
        FinalizeCapture();
        return;
    }

    if (!m_pCurrentSpawnedActor)
    {
        SpawnCurrentActor();
        return;
    }

    // ---- Local helpers ----
    auto ResetLightAndCamera = [this]()
        {
            m_iCurrentLightColorIndex = 0;
            m_iCurrentCameraIndex = 0;
        };

    auto HideFog = [this]()
        {
            if (m_pLocalFog) m_pLocalFog->SetActorHiddenInGame(true);
            m_bFogActive = false;
            m_pMetadataWriter->setIsFogEnabled(false);
        };

    auto ShowFog = [this]()
        {
            if (m_pLocalFog) m_pLocalFog->SetActorHiddenInGame(false);
            m_bFogActive = true;
            m_pMetadataWriter->setIsFogEnabled(true);
        };

    // ---- Main state machine ----
    // camera
    if (m_iCurrentCameraIndex < m_aCameraTargets.Num())
    {
        CaptureScreenshotForCurrentCamera();
        return;
    }

    // light color
    ++m_iCurrentLightColorIndex;
    if (m_aLightColors.IsValidIndex(m_iCurrentLightColorIndex))
    {
        SetAllLightsColor(m_aLightColors[m_iCurrentLightColorIndex]);
        m_iCurrentCameraIndex = 0;
        CaptureScreenshotForCurrentCamera();
        return;
    }

    // fog
    if (m_bAddFog && !m_bFogActive)
    {
        ShowFog();
        ResetLightAndCamera();
        if (m_aLightColors.IsValidIndex(0))
            SetAllLightsColor(m_aLightColors[0]);
        CaptureScreenshotForCurrentCamera();
        return;
    }

    // materials
    if (m_aMaterials.IsValidIndex(m_iCurrentMaterialIndex))
    {
        ApplyCurrentMaterial();
        ++m_iCurrentMaterialIndex;

        HideFog();
        ResetLightAndCamera();
        if (m_aLightColors.IsValidIndex(0))
            SetAllLightsColor(m_aLightColors[0]);
        CaptureScreenshotForCurrentCamera();
        return;
    }

    // Clean up for current actor
    if (!m_pCurrentSpawnedActor->IsPendingKillPending())
    {
        m_pCurrentSpawnedActor->Destroy();
    }
    HideFog();

    m_pCurrentSpawnedActor = nullptr;
    ++m_iCurrentActorIndex;

    // reset iteration indices
    m_iCurrentMaterialIndex = 0;
    ResetLightAndCamera();

    // process the next actor
    ProcessCaptureState();
}

// ============================================================================
// Actor spawning
// ============================================================================
void UDatasetCaptureManager::SpawnCurrentActor()
{
    const TPair<TSubclassOf<AActor>, int32>& Entry = m_aActorEntries[m_iCurrentActorIndex];
    const TSubclassOf<AActor>                ActorClass = Entry.Key;

    if (!ensureAlwaysMsgf(ActorClass, TEXT("Invalid ActorClass at index %d"), m_iCurrentActorIndex))
    {
        ++m_iCurrentActorIndex;
        ProcessCaptureState();
        return;
    }

    const FTransform SpawnXf = m_pObjectTarget
        ? m_pObjectTarget->GetActorTransform()
        : FTransform::Identity;

    m_pCurrentSpawnedActor = m_pWorld->SpawnActor<AActor>(ActorClass, SpawnXf);
    if (!ensureAlwaysMsgf(m_pCurrentSpawnedActor,
        TEXT("Failed to spawn actor of class %s"), *ActorClass->GetName()))
    {
        ++m_iCurrentActorIndex;
        ProcessCaptureState();
        return;
    }

    UE_LOG(LogTemp, Log, TEXT("Capture Manager: Spawned %s"), *m_pCurrentSpawnedActor->GetName());

    BuildCameraTargetsForCurrentActor();

    // metadata
    m_pMetadataWriter->setModelName(ActorClass->GetName());
    m_pMetadataWriter->setMaterialName(TEXT("Default"));

    // Ensure correct lights and metadata for first pass
    m_iCurrentCameraIndex = 0;
    m_iCurrentLightColorIndex = 0;
    if (m_aLightColors.IsValidIndex(0))
    {
        SetAllLightsColor(m_aLightColors[0]);
    }

    ProcessCaptureState();
}

// ============================================================================
// Screenshot capture
// ============================================================================
void UDatasetCaptureManager::CaptureScreenshotForCurrentCamera()
{
    static constexpr float ScreenshotDelay = 0.05f; // seconds

    const FVector& CamLoc = m_aCameraTargets[m_iCurrentCameraIndex];

    if (APlayerController* PC = m_pWorld->GetFirstPlayerController())
    {
        if (APawn* Pawn = PC->GetPawn())
        {
            const FRotator LookAt =
                UKismetMathLibrary::FindLookAtRotation(CamLoc, m_vCurrentObjectCenter);

            Pawn->SetActorLocation(CamLoc);
            PC->SetControlRotation(LookAt);
        }
    }

    m_pMetadataWriter->setCameraPosition(m_vCameraTransforms[m_iCurrentCameraIndex].ToString());

    /* ------------------------------------------------------------------ */
    const int32   ClassIndex = m_aActorEntries[m_iCurrentActorIndex].Value;
    const FString MapName = m_pWorld->GetMapName();
    const FString DatasetRootRel = FPaths::Combine(TEXT("Dataset"), FString::FromInt(ClassIndex), MapName);
    const FString DatasetRootAbs = FPaths::Combine(FPaths::ProjectDir(), DatasetRootRel);

    IPlatformFile& PF = FPlatformFileManager::Get().GetPlatformFile();
    PF.CreateDirectoryTree(*DatasetRootAbs);

    const auto MakeScreenshotName = [&](const FString& Dir) -> FString
        {
            return FString::Printf(
                TEXT("%s/%s%s_%d%d%d%d.png"),
                *Dir,
                *MapName,
                *m_pCurrentSpawnedActor->GetName(),
                m_iCurrentCameraIndex,
                m_iCurrentLightColorIndex,
                m_iCurrentMaterialIndex,
                static_cast<int32>(m_bFogActive));
        };

    m_sCurrentScreenshotPath = MakeScreenshotName(DatasetRootAbs);
    m_sRelativeImagePath = MakeScreenshotName(DatasetRootRel);

    m_pMetadataWriter->setClassIndex(FString::FromInt(ClassIndex));

    // Fire the screenshot after a tiny delay to ensure the next frame was rendered.
    FTimerHandle Handle;
    m_pWorld->GetTimerManager()
        .SetTimer(Handle, this,
            &UDatasetCaptureManager::RequestCameraScreenshot,
            ScreenshotDelay, false);
}

// ============================================================================
// Request a screenshot of the current camera
// ============================================================================
void UDatasetCaptureManager::RequestCameraScreenshot()
{
    UGameViewportClient* Viewport = m_pWorld ? m_pWorld->GetGameViewport() : nullptr;
    if (!Viewport)
    {
        UE_LOG(LogTemp, Warning, TEXT("Capture Manager: No GameViewportClient found."));
        ++m_iCurrentCameraIndex;
        ProcessCaptureState();
        return;
    }

    // Ensure only one delegate
    if (m_oScreenshotCapturedHandle.IsValid())
    {
        Viewport->OnScreenshotCaptured().Remove(m_oScreenshotCapturedHandle);
        m_oScreenshotCapturedHandle.Reset();
    }

    m_oScreenshotCapturedHandle = Viewport->OnScreenshotCaptured()
        .AddUObject(this, &UDatasetCaptureManager::OnScreenshotCaptured);

    FScreenshotRequest::RequestScreenshot(m_sCurrentScreenshotPath, /*bShowUI=*/false, /*bAddFilenameSuffix=*/false);
}

// ============================================================================
// Helper to spawn a temporary local-fog volume
// ============================================================================
ALocalFogVolume* UDatasetCaptureManager::CreateTempFogVolume(
    const FVector&       Location,
    const FVector&       UniformScale,
    float                RadialDensity,
    float                HeightDensity,
    const FLinearColor&  Albedo,
    float                PhaseG)
{
    if (!m_pWorld) return nullptr;

    FActorSpawnParameters Params;
    Params.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;

    ALocalFogVolume* FogActor =
        m_pWorld->SpawnActor<ALocalFogVolume>(Location, FRotator::ZeroRotator, Params);
    if (!FogActor) return nullptr;

    // convert cm -> m
    constexpr float UnitsPerMetre = 100.f;
    FogActor->SetActorScale3D(UniformScale / UnitsPerMetre);

    if (ULocalFogVolumeComponent* Fog = FogActor->FindComponentByClass<ULocalFogVolumeComponent>())
    {
        Fog->SetRadialFogExtinction(RadialDensity);
        Fog->SetHeightFogExtinction(HeightDensity);
        Fog->SetFogAlbedo(Albedo);
        Fog->SetFogPhaseG(PhaseG);
        Fog->SetVisibility(true);
        Fog->RegisterComponent();
    }
    return FogActor;
}


// ============================================================================
// Apply a colour to every ALight in the level
// ============================================================================
void UDatasetCaptureManager::SetAllLightsColor(const FLinearColor& NewColor) const
{
    if (!m_pWorld) return;

    TArray<AActor*> Lights;
    UGameplayStatics::GetAllActorsOfClass(m_pWorld, ALight::StaticClass(), Lights);

    for (AActor* Actor : Lights)
    {
        if (ALight* Light = Cast<ALight>(Actor))
        {
            if (ULightComponent* LC = Light->GetLightComponent())
            {
                LC->SetLightColor(NewColor);
            }
        }
    }

    if (m_pMetadataWriter)
    {
        m_pMetadataWriter->setLightColor(NewColor.ToString());
    }
}

// ============================================================================
// Apply a material to every mesh slot on the current actor
// ============================================================================
void UDatasetCaptureManager::ApplyCurrentMaterial()
{
    if (!m_pCurrentSpawnedActor ||
        !m_aMaterials.IsValidIndex(m_iCurrentMaterialIndex))
    {
        return;
    }

    UMaterialInterface* Mat = m_aMaterials[m_iCurrentMaterialIndex];
    if (!Mat) return;

    TArray<UMeshComponent*> Meshes;
    m_pCurrentSpawnedActor->GetComponents<UMeshComponent>(Meshes);

    for (UMeshComponent* Mesh : Meshes)
    {
        const int32 NumSlots = Mesh->GetNumMaterials();
        for (int32 Slot = 0; Slot < NumSlots; ++Slot)
        {
            Mesh->SetMaterial(Slot, Mat);
        }
    }

    if (m_pMetadataWriter)
    {
        m_pMetadataWriter->setMaterialName(Mat->GetName());
    }
}

// ============================================================================
// Screenshot callback
// ============================================================================
void UDatasetCaptureManager::OnScreenshotCaptured(
    int32            Width,
    int32            Height,
    const TArray<FColor>& Bitmap)
{
    /* Always unhook the delegate first */
    if (UGameViewportClient* VP = m_pWorld ? m_pWorld->GetGameViewport() : nullptr)
    {
        VP->OnScreenshotCaptured().Remove(m_oScreenshotCapturedHandle);
    }
    m_oScreenshotCapturedHandle.Reset();

    /* Save PNG & write metadata */
    if (!m_sCurrentScreenshotPath.IsEmpty() &&
        Bitmap.Num() == Width * Height &&
        SavePNG(*m_sCurrentScreenshotPath, Bitmap, Width, Height))
    {
        UE_LOG(LogTemp, Log, TEXT("Capture Manager: Screenshot saved: %s"), *m_sCurrentScreenshotPath);

        if (m_pMetadataWriter)
        {
            m_pMetadataWriter->setImageName(m_sRelativeImagePath);
            m_pMetadataWriter->WriteToFile();
        }
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("Capture Manager: Screenshot save FAILED: %s"), *m_sCurrentScreenshotPath);
    }

    /* Advance the state machine */
    ++m_iCurrentCameraIndex;
    ProcessCaptureState();
}

// ============================================================================
// Finish a full dataset pass (optionally open the next level)
// ============================================================================
void UDatasetCaptureManager::FinalizeCapture()
{
    UE_LOG(LogTemp, Log, TEXT("Capture Manager: All Actors processed!"));

    if (!m_pNextLevel.IsNull())
    {
        const FName LevelName(*m_pNextLevel.ToSoftObjectPath().GetLongPackageName());
        if (!LevelName.IsNone())
        {
            UE_LOG(LogTemp, Log, TEXT("Capture Manager: Loading next level: %s"), *LevelName.ToString());
            UGameplayStatics::OpenLevel(m_pWorld, LevelName);
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("Capture Manager: NextLevel asset had an empty package name."));
        }
    }
    else
    {
        UE_LOG(LogTemp, Log, TEXT("Capture Manager: No next level specified  staying put."));
    }

    RemoveFromRoot();   // allow GC once the manager finishes
}

// ============================================================================
// Raw -> PNG helper
// ============================================================================
bool UDatasetCaptureManager::SavePNG(
    const FString& Filename,
    const TArray<FColor>& SrcBitmap,
    int32                 Width,
    int32                 Height)
{
    IImageWrapperModule& ImageWrapper =
        FModuleManager::LoadModuleChecked<IImageWrapperModule>("ImageWrapper");

    TSharedPtr<IImageWrapper> Png = ImageWrapper.CreateImageWrapper(EImageFormat::PNG);
    if (Png.IsValid() &&
        Png->SetRaw(
            SrcBitmap.GetData(),
            SrcBitmap.GetAllocatedSize(),
            Width, Height,
            ERGBFormat::BGRA, 8))
    {
        const TArray64<uint8>& Data = Png->GetCompressed(100);   // max quality
        return FFileHelper::SaveArrayToFile(Data, *Filename);
    }
    return false;
}

// ============================================================================
// Build camera locations that perfectly frame the current actor
// ============================================================================
void UDatasetCaptureManager::BuildCameraTargetsForCurrentActor()
{
    if (!m_pCurrentSpawnedActor || !m_pWorld) return;

    // Bounding sphere of the actor
    const FBox  Bounds = m_pCurrentSpawnedActor->GetComponentsBoundingBox(true);
    m_vCurrentObjectCenter = Bounds.GetCenter();
    m_fCurrentObjectRadius = Bounds.GetExtent().Size();

    // Current camera FoV
    float CamFOVdeg = 50.f;
    if (APlayerController* PC = m_pWorld->GetFirstPlayerController())
    {
        if (PC->PlayerCameraManager)
            CamFOVdeg = PC->PlayerCameraManager->GetFOVAngle();
    }

    // position actor inside the inner 256 px of 512 of the frame
    constexpr float CropFraction = 256.f / 512.f;    
    const float HalfFOVradCrop =
        FMath::DegreesToRadians(CamFOVdeg * 0.5f * CropFraction);

    const float CamDistance =
        m_fCurrentObjectRadius / FMath::Tan(HalfFOVradCrop) * 1.05f; // 5 % margin

    m_aCameraTargets.Reset();

    const FTransform Ref =
        m_pObjectTarget ? m_pObjectTarget->GetActorTransform() : FTransform::Identity;

    for (const FVector& Dir : m_vCameraTransforms)
    {
        const FVector WorldDir = Ref.GetRotation().RotateVector(Dir.GetSafeNormal());
        m_aCameraTargets.Add(m_vCurrentObjectCenter + WorldDir * CamDistance);
    }

    m_iCurrentCameraIndex = 0;  // reset camera index
}

