COMMENT

"Kv3.2" channel.
BBP Kv channel, but activation/inactivation vHalf and slope adjusted to fit experimental data.
Use this mechanism to 'mutate' given proportion of Kv3.2 channels.

ENDCOMMENT

:Reference : :		Characterization of a Shaw-related potassium channel family in rat brain, The EMBO Journal, vol.11, no.7,2473-2486 (1992)

NEURON	{
	SUFFIX SKv3_1m
	USEION k READ ek WRITE ik
	RANGE gSKv3_1bar, gSKv3_1, ik 
	RANGE a,b,c,d,e
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gSKv3_1bar = 0.00001 (S/cm2)
	a = -20.0			 (1)
    b = -10.2			 (1)
    c = 4 * 7			 (1)
    d = -10.0			 (1)
    e = 6.0				 (1)
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
		mInf =  1/(1+exp(((v -(a))/(b))))
		mTau =  c/(1+exp(((v -(d))/(e))))
	UNITSON
}

