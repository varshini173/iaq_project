// AUTO-GENERATED. Inputs, in order: T, RH, PT08.S1(CO), PT08.S2(NMHC), PT08.S3(NOx), PT08.S4(NO2), PT08.S5(O3)
#pragma once
#define IAQ_N_FEATURES 7
static const float iaq_feature_mean[IAQ_N_FEATURES] = {17.515499f, 47.619743f, 1114.374146f, 952.236328f, 804.288696f, 1415.448242f, 1056.702148f};
static const float iaq_feature_std[IAQ_N_FEATURES]  = {9.320841f, 17.460300f, 212.529129f, 262.432007f, 238.621094f, 364.536774f, 409.857056f};
static const char* iaq_class_names[3] = {"Good", "Moderate", "Poor"};
