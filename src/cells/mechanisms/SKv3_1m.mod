:Comment : "Kv3.2" channel
:Comment : BBP Kv channel, but activation/inactivation vHalf and slope adjusted to fit experimental data.
:Comment : Use this mechanism to 'mutate' given proportion of Kv3.2 channels.

:Reference : Characterization of a Shaw-related potassium channel family in rat brain, The EMBO Journal, vol.11, no.7,2473-2486 (1992)

:Comment: Adapted by Christopher Currin (2022) to match experimental data for channel in HEK cells from Ethan GOldberg's lab

NEURON	{
	SUFFIX SKv3_1m
	USEION k READ ek WRITE ik
	RANGE gSKv3_1bar, gSKv3_1, ik 
	RANGE iv_shift, iv_gain, tau_scale, tau_shift, tau_gain
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gSKv3_1bar = 0.00001 (S/cm2)
	iv_shift = -20.0			 (1)
    iv_gain = -10.2			 	 (1)
    tau_scale = 28				 (1)
    tau_shift = -10.0			 (1)
    tau_gain = 6.0				 (1)
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	gSKv3_1	(S/cm2)
	mInf
	mTau
}

STATE	{ 
	m
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gSKv3_1 = gSKv3_1bar*m
	ik = gSKv3_1*(v-ek)
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau
}

INITIAL{
	rates()
	m = mInf
}

PROCEDURE rates(){
	UNITSOFF
		mInf =  1/(1+exp(((v -(iv_shift))/(iv_gain))))
		mTau =  tau_scale/(1+exp(((v -(tau_shift))/(tau_gain))))
	UNITSON
}

