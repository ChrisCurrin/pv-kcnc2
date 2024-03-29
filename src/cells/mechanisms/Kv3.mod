TITLE fast activated potassium Kv3 (Kv3.1/3.2) for PV interneurons

COMMENT
 modeled by Gunay et al., 2008
 implemented in NEURON by Kitano, 2011
ENDCOMMENT

UNITS {
 (mV) = (millivolt)
 (mA) = (milliamp)
}

NEURON {
 SUFFIX Kv3
 USEION k READ ek WRITE ik
 RANGE gmax, iKv3, ik
 RANGE theta_m, k_m, tau_m0, tau_m1, phi_m0, phi_m1, sigma_m0, sigma_m1,  h0, theta_h, k_h, tau_h0, tau_h1, phi_h, sigma_h0, sigma_h1
}

PARAMETER {
 v (mV)
 gmax  = 0.001 (mho/cm2)
 theta_m = -13.0 (mV)
 k_m = 7.8 (mV)
 tau_m0 = 0.1 (ms)
 tau_m1 = 80.0 (ms)
 phi_m0 = -20.0 (mV)
 phi_m1 = -13.0 (mV)
 sigma_m0 = 60.0 (mV)
 sigma_m1 = -6.0 (mV)
 h0 = 0.95 
 theta_h = -20.0 (mV)
 k_h = -10.0 (mV)
 tau_h0 = 33.0 (ms)
 tau_h1 = 33.0 (ms)
 phi_h = 0.0 (mV)
 sigma_h0 = 10.0 (mV)
 sigma_h1 = -10.0 (mV)
}

STATE {
 m h
}

ASSIGNED { 
 ik (mA/cm2)
 iKv3 (mA/cm2)
 ek (mV)
 minf
 taum (ms)
 hinf
 tauh (ms)
}

BREAKPOINT {
 SOLVE states METHOD cnexp
 ik  = gmax*m*m*m*m*h*(v-ek)
 iKv3 = ik
}

UNITSOFF

INITIAL {
 settables(v)
 m = minf
 h = hinf
}

DERIVATIVE states {  
 settables(v)
 m' = (minf - m)/taum
 h' = (hinf - h)/tauh
}

PROCEDURE settables(v) {
    TABLE minf, taum, hinf, tauh FROM -100 TO 100 WITH 400

	minf = 1.0 / (1.0 + exp((theta_m - v)/k_m))
	taum = tau_m0 + (tau_m1 - tau_m0)/(exp((phi_m0 - v)/sigma_m0) + exp((phi_m1 - v)/sigma_m1))
	hinf = h0 + (1.0 - h0) / (1.0 + exp((theta_h - v)/k_h))
	tauh = tau_h0 + (tau_h1 - tau_h0)/(exp((phi_h - v)/sigma_h0) + exp((phi_h - v)/sigma_h1))
}

UNITSON






