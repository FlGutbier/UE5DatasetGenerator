// Fill out your copyright notice in the Description page of Project Settings.

#include "DatasetMetadataWriter.h"

/***********************************************************************************************/
void UDatasetMetadataWriter::Initialize()
{
    m_sLevelName = TEXT("");
    m_sMaterialName = TEXT("");
    m_sLightColor = TEXT("");
    m_sCameraPosition = TEXT("");
    m_sImageName = TEXT("");
    m_sFilePath = TEXT("");
}

/***********************************************************************************************/
void UDatasetMetadataWriter::CreateFile() const
{
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();

    // Write header
    if (!PlatformFile.FileExists(*m_sFilePath))
    {
        FString Header = TEXT("Image,Object,Level,Camera Position\n");
        FFileHelper::SaveStringToFile(Header, *m_sFilePath, FFileHelper::EEncodingOptions::AutoDetect);
    }
}

/***********************************************************************************************/
void UDatasetMetadataWriter::WriteToFile() const
{
    FString LineToWrite = FString::Printf(TEXT("%s,%s,%s,%s,%s\n"),
        *m_sImageName,
        *m_sModelName,
        *m_sLevelName,
        *m_sCameraPosition,
        *m_sLightColor);

    FFileHelper::SaveStringToFile(LineToWrite, *m_sFilePath, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM, &IFileManager::Get(), FILEWRITE_Append);
}
