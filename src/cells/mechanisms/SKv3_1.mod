:Comment : "Kv3.2" channel
:Reference : :		Characterization of a Shaw-related potassium channel family in rat brain, The EMBO Journal, vol.11, no.7,2473-2486 (1992)

NEURON	{
	SUFFIX SKv3_1
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
	iv_shift = -2.11 			(1)
	iv_gain = -9.7 				(1)
	tau_scale = 20.0 			(1)
	tau_shift = 10.0 			(1)
	tau_gain = 8	 			(1)
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

