// Copyright (c) 2025 Florian Gutbier
// 
// This source code is part of the UE5 Plugin developed for the Bachelor's thesis
// at the University of Bamberg.
// 
// Released under the MIT License. See LICENSE file for details.

#include "DatasetMetadataWriter.h"

/***********************************************************************************************/
void UDatasetMetadataWriter::Initialize()
{
    m_sLevelName = TEXT("");
    m_sModelName = TEXT("");
    m_sMaterialName = TEXT("");
    m_sLightColor = TEXT("");
    m_sCameraPosition = TEXT("");
    m_sImageName = TEXT("");
    m_sFilePath = TEXT("");
    m_sIsFogEnabled = TEXT("false");
    m_sClassIndex = TEXT("");
}

/***********************************************************************************************/
void UDatasetMetadataWriter::CreateFile() const
{
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();

    // Write header
    if (!PlatformFile.FileExists(*m_sFilePath))
    {
        FString Header = TEXT("Image;Object;Level;Class;Material;Camera Position;Light Color (RGB);Fog\n");
        FFileHelper::SaveStringToFile(Header, *m_sFilePath, FFileHelper::EEncodingOptions::AutoDetect);
    }
}

/***********************************************************************************************/
void UDatasetMetadataWriter::WriteToFile() const
{
    FString LineToWrite = FString::Printf(TEXT("%s;%s;%s;%s;%s;%s;%s;%s\n"),
        *m_sImageName,
        *m_sModelName,
        *m_sLevelName,
        *m_sClassIndex,
        *m_sMaterialName,
        *m_sCameraPosition,
        *m_sLightColor,
        *m_sIsFogEnabled);

    FFileHelper::SaveStringToFile(LineToWrite, *m_sFilePath, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM, &IFileManager::Get(), FILEWRITE_Append);
}
