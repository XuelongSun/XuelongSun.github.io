---
layout: post
title:  "Neuron Models"
date:   2023-04-18 12:42:00 +0800
tags: Web Neuroscience Tutorial
author: Xuelong Sun
---

The models in this category describe the relationship between neuronal membrane currents at the input stage, and membrane voltage at the output stage. This category includes (generalized) integrate-and-fire models and biophysical models inspired by the work of Hodgkin–Huxley in the early 1950s using an experimental setup that punctured the cell membrane and allowed to force a specific membrane voltage/current ([see wiki](https://en.wikipedia.org/wiki/Biological_neuron_mode)). 

#### 1. Hodgkin-Huxley neuron model (HH model)
$$C_m\frac{dV_m(t)}{dt} = I - I_{ion}$$

$$I_{ion} = g_{Na}m^3h(V_m - E_{Na}) - g_Kn^4(V_m - E_k) - g_l(V_m - E_l)$$

where, the $m$ (quick Na+ influx), $n$ (slow K+ outflux), $$h$$ (slow Na+ influx) represents the voltage gated ionic channels ( dimensionless probabilities between 0 and 1 that are associated with channel subunits):

$$\frac{dx}{dt} = \alpha(V_m)(1 - x) - \beta(V_m)x$$

where $$x = {m, n, h}$$ and $$\alpha$$ and $$\beta$$ refers to the time constant of gate open and close respectively.

This differential equation have the following solution when $$V_m$$ is constant:

$$x(t) = x(0)e^{-(\alpha + \beta)t} + \frac{\alpha}{\alpha + \beta}(1 - e^{-(\alpha+\beta)t}), t \geq 0$$

the quantity $$\tau = 1 / (\alpha + \beta)$$ is the time constant of the rate process at constant membrane potential. The dimensionless quantity $$ \alpha / (\alpha + \beta) = \tau\alpha$$  is the steady-state probability at constant membrane potential.

$$\alpha_m(V_m) = \frac{0.1(25-V)}{e^{\frac{25-V}{10}}-1}$$,
$$\alpha_n(V_m) = \frac{0.01(10-V)}{e^{\frac{10-V}{10}}-1}$$,
$$\alpha_h(V_m) = 0.07e^{-V/20}$$,

$$\beta_m(V_m) = 4e^{-V/18}$$,
$$\beta_n(V_m) = 0.125e^{-V/80}$$,
$$\beta_h(V_m) = \frac{1}{e^{\frac{30-V}{10}}+1}$$,

where $$V = V_m - V_{rest}$$ denots the depolarization.

```python
class HHNeuron:
    class Gate:
        # this are dynamically updateds
        alpha, beta, state = 0, 0, 0
        
        def update(self, deltaTms):
            alphaState = self.alpha*(1 - self.state)
            betaState = self.beta * self.state
            self.state += deltaTms * (alphaState - betaState)
        
        def setInfiniteState(self):
            self.state = self.alpha / (self.alpha + self.beta)
    
    gNa, gK, gKleak = 120, 36, 0.3
    m, n, h = Gate(), Gate(), Gate()
    Cm = 1 # uF/cm^2
    
    def __init__(self, startingVoltage=-65):
        self.ENa, self.EK, self.EKleak = 115+startingVoltage, -12+startingVoltage, 10.6+startingVoltage
        self.Vrest = startingVoltage
        self.Vm = self.Vrest
        self.UpdateGateTimeConstants(startingVoltage)
        self.m.setInfiniteState()
        self.n.setInfiniteState()
        self.n.setInfiniteState()
    
    def UpdateGateTimeConstants(self, Vm):
        """Update time constants of all gates based on the given Vm"""
        self.n.alpha = .01 * ((10-(Vm - self.Vrest)) / (np.exp((10-(Vm - self.Vrest))/10)-1))
        self.n.beta = .125*np.exp(-(Vm - self.Vrest)/80)
        self.m.alpha = .1*((25-(Vm - self.Vrest)) / (np.exp((25-(Vm - self.Vrest))/10)-1))
        self.m.beta = 4*np.exp(-(Vm - self.Vrest)/18)
        self.h.alpha = .07*np.exp(-(Vm - self.Vrest)/20)
        self.h.beta = 1/(np.exp((30-(Vm - self.Vrest))/10)+1)

    def UpdateCellVoltage(self, stimulusCurrent, deltaTms):
        """calculate channel currents using the latest gate time constants"""
        INa = np.power(self.m.state, 3) * self.gNa * self.h.state*(self.Vm-self.ENa)
        IK = np.power(self.n.state, 4) * self.gK * (self.Vm-self.EK)
        IKleak = self.gKleak * (self.Vm-self.EKleak)
        Isum = stimulusCurrent - INa - IK - IKleak
        self.Vm += deltaTms * Isum / self.Cm

    def UpdateGateStates(self, deltaTms):
        """calculate new channel open states using latest Vm"""
        self.n.update(deltaTms)
        self.m.update(deltaTms)
        self.h.update(deltaTms)

    def Iterate(self, stimulusCurrent=0, deltaTms=0.05):
        self.UpdateGateTimeConstants(self.Vm)
        self.UpdateCellVoltage(stimulusCurrent, deltaTms)
        self.UpdateGateStates(deltaTms)
```
To run simulation and plot the results:
```python
hh = HHNeuron()
pointCount = 5000
voltages = np.empty(pointCount)
times = np.arange(pointCount) * 0.05
stim = np.zeros(pointCount)
stim[1200:3800] = 6.5  # create a square pulse

for i in range(len(times)):
    hh.Iterate(stimulusCurrent=stim[i], deltaTms=0.05)
    voltages[i] = hh.Vm
    # note: you could also plot hh's n, m, and k (channel open states)

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6),
                             gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(times, voltages, 'b')
ax1.set_ylabel("Membrane Potential (mV)")
ax1.set_title("Hodgkin-Huxley Spiking Neuron Model", fontsize=16)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.tick_params(bottom=False)

ax2.plot(times, stim, 'r')
ax2.set_ylabel("Stimulus (µA/cm²)")
ax2.set_xlabel("Simulation Time (milliseconds)")
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.margins(0, 0.1)
plt.tight_layout()

```

![](/assets/img/posts/HH_output.png)

#### 1.2 Integrate-and-fire model (LIF)

In the most general form:

$$C_m\frac{dV}{dt} = g_mf(V) + I + I_a$$

Perfect Integrate-and-fire model (LIF)

$$C_m\frac{dV}{dt} = I$$

Leaky Integrate-and-fire model (LIF)

$$C_m\frac{dV}{dt} = -g_L(V - E_L) + I$$
if $$ V(t) = V_{th} $$ then $$V(t + \Delta t) = E_L$$

+ subthreshold current step: $$V(t) = (\frac{I}{g_L} + E_L)(1 - e^{-t/\tau_m})$$, $$\tau_m = C_m/g_L$$
+ suprathreshold current step: regular firing

Adaptive Integrate-and-fire model

$$C_m\frac{dV}{dt} = -g_L(V - E_L) + I - \sum_k{w_k}$$
if $$ V(t) = V_{th} $$ then $$V(t + \Delta t) = E_L$$

$$ \tau_k\frac{dw_k}{dt} = a_k(V - E_L) - w_k$$

add linearized spike-triggered current
$$w_k(t + \Delta t) = w_k + b_k$$

to enable spike frequency adaption and is equivalent to HH model.

Further, adding exponential/quadratic function in the vlotage-dependent leak:

$$C_m\frac{dV}{dt} = -g_L(V - E_L) + \Delta_{T}e^{\frac{V - V_{rh}}{\Delta_T}}/R + I - \sum_k{w_k}$$

```python
class LIFNeuron:
    def __init__(self, dt=0.1):
        self.v_th = -55.0  # mv
        self.v_reset = -75.0 # mv
        self.tau = 10.0 # ms
        self.gL = 10.0 # msi
        self.EL = -75.0 # mv
        self.v = -75.0 # mv
        
        # refractory period
        self.rf_t_p = 4.0
        self.rf_t = 0
        
        self.spike = 0
        
        self.dt = dt # ms
    
    def update(self, I):
        self.spike = 0
        if self.rf_t > 0:
            # still in refractory period
            self.rf_t -= self.dt
        elif self.v >= self.v_th:
            # action potential
            self.spike = 1
            self.v = self.v_reset
            self.rf_t = self.rf_t_p
        else:
            # update membrane potential
            self.v += (-(self.v - self.EL) + I / self.gL) * self.dt / self.tau
        
        return self.v, self.spike
            
    
    def run(self, timeout, I):
        tT = int(timeout/self.dt)
        
        r_spikes = []
        r_v = np.zeros([tT])
        
        # re-assign injection current
        if isinstance(I, float):
            # constant injecting current
            inj_I = np.ones(tT) * I
        elif len(I) == 1:
            # pulse current
            inj_I = np.ones(tT) * I
            inj_I[:int(tT/2)-1000] = 0
            inj_I[int(tT/2)+1000:] = 0
        elif len(I) < tT:
            print('not valid I input')
            return 1
        
        for t in range(tT):
            r_v[t] = self.v
            if self.spike == 1:
                r_spikes.append(t)
            self.update(inj_I[t])
        
        return r_v, r_spikes
```
run simulation and plot the results

```python
neuron = LIFNeuron(dt=0.1)

v1, sp1 = neuron.run(500, [260])

neuron.v = -75
v2, sp2 = neuron.run(500, [130])

fig, ax = plt.subplots(figsize=(12,6))

ax.plot(v1)
ax.scatter(sp1, len(sp1)*[-54], color='k', marker='^')
ax.plot(v2)
```

![](/assets/img/posts/LIF_output.png)

##### 1.3 Spike response model
Also the IF type of model with a firing threshold, but more general than leak IF

$$V(t) = \eta(t - \hat{t}) + \int_{0}^{+\infty}\kappa(t - \hat{t}, s)I(t - s)ds$$

if $$V(t) \geq \theta$$ and $$\dot{V(t)} > 0$$, then $$\hat{t} = t$$, where $$\theta$$ is the threshold that can be time-dependent:

$$\theta(t - \hat{t}) = +\infty$$ if $$t - \hat{t} \leq \gamma_{ref}$$ else $$\theta_0 + \theta_1e^{-(t - \hat{t})/\tau}$$,

where $$\gamma_{ref}$$ is a fixed absolute refractory period. 