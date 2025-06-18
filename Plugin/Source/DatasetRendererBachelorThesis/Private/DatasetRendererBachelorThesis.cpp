// Copyright (c) 2025 Florian Gutbier
// 
// This source code is part of the UE5 Plugin developed for the Bachelor's thesis
// at the University of Bamberg.
// 
// Released under the MIT License. See LICENSE file for details.

#include "DatasetRendererBachelorThesis.h"

#define LOCTEXT_NAMESPACE "FDatasetRendererBachelorThesisModule"

void FDatasetRendererBachelorThesisModule::StartupModule()
{
	// This code will execute after your module is loaded into memory; the exact timing is specified in the .uplugin file per-module
	
}

void FDatasetRendererBachelorThesisModule::ShutdownModule()
{
	// This function may be called during shutdown to clean up your module.  For modules that support dynamic reloading,
	// we call this function before unloading the module.
	
}

#undef LOCTEXT_NAMESPACE
	
IMPLEMENT_MODULE(FDatasetRendererBachelorThesisModule, DatasetRendererBachelorThesis)
