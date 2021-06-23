#imports
from galpy.orbit import Orbit
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.constants as c
import astropy.units as u
from galpy.potential import LogarithmicHaloPotential
from galpy import potential as gp
from galpy.util import bovy_conversion
import time as pytime
import sys

#GALPY scaling variables
ro=8.
vo=220.
to=bovy_conversion.time_in_Gyr(ro=ro,vo=vo)
mo=bovy_conversion.mass_in_msol(ro=ro,vo=vo)

#integration time
torb = np.linspace(0,12.,250)

#define a probability density function that will be used to generate subhalo masses
def rndm(a, b, g, size=1):
    """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b
    """
    r = np.random.random(size=size)
    ag, bg = a**g, b**g
    return (ag + (bg - ag)*r)**(1./g)

# define a list to be used later so that the output is easily readable 
  all_info=[' mean total subhalos',' total subhalo disp',' mean mass differene',' mass diff disp',' mean 10e6 subhalos',' 10e6 disp',' mean 10e7 subhalos',' 10e7 disp',' mean 10e8 subhalos',' 10e8 disp',' vx disp',' vy disp',' vz disp',' mean r',' r disp',' E disp']

#set the substructure mass fraction
smf = 0.03

#potential creation (total,smooth,clumpy)
tp = gp.LogarithmicHaloPotential(amp=1., ro=ro,vo=vo)
sp = gp.LogarithmicHaloPotential(amp=1.-smf, ro=ro,vo=vo)
cp = gp.LogarithmicHaloPotential(amp=smf, ro=ro,vo=vo)

#radius and mass at each radius in each potential
r=np.linspace(0,100.,5000)
tmar = []
smar = []
cmar = []
for i in r:
    tmar.append(tp.mass(i/ro))
    smar.append(sp.mass(i/ro))
    cmar.append(cp.mass(i/ro))
    
#define a function that will integrate a tracer particle(s) in a generated 'random' subhalo population, and save a list of information associated with both the subhalo population and the tracer
def substructure_integration(num_scenarios=int,tracer_r=float or int or list,sub_halo_mass=None):
    '''
    tracer_r as a float or int will integrate a single tracer at that radius.
    tracer_r as a list will integrate len(tracer_r) tracer particles at each list value radius.
    
    num_scenarios defines the number of integrations to be done for each tracer particle. For each new scenario, a new subhalo population is generated
    
    when sub_halo_mass is None a mass spectrum population of subhalos is generated with masses between 10^6 and 10^9 solar masses. Otherwise, every subhalo will take the exact mass passed to sub_halo_mass
    '''
    
    #initialize the tracer orbit as circular in the smooth potential (total - sub potential)
    if type(tracer_r)==int or type(tracer_r)==float:
        tracer_r=[tracer_r]

    tracer_r = np.array(tracer_r)
    tracer_cv = sp.vcirc(tracer_r)
    tracer_zeros = np.zeros(len(tracer_r))
    tracerorb = np.column_stack([tracer_r/ro,tracer_zeros,tracer_cv/vo,tracer_zeros,tracer_zeros,tracer_zeros])
        
    #initialize arrays for output
    output_per_scenario=[]
    energy_evolution_per_scenario=np.zeros((num_scenarios,len(tracer_r),len(torb)))
    radius_evolution_per_scenario=np.zeros((num_scenarios,len(tracer_r),len(torb)))

    #redo everything in the loop below the number of specified times
    for ns in range(num_scenarios):

        #print the scenario number that the loop is on
        print('Started scenario '+str(ns+1))
        
        #generate subhalo masses
        if sub_halo_mass is None:
            #generate random subhalo masses
            masses = rndm(10**6,10**9,-1,size=1000000)
        else:
            masses=np.ones(int(cmar[-1]/sub_halo_mass))*sub_halo_mass
            
        #determine how many subhalos will make up the distribution
        cum_masses=np.cumsum(masses)
        indx=cum_masses<cmar[-1]
        count=np.sum(indx)
        total_sh_mass=np.sum(masses[indx])
        sh_masses = masses[0:count]
        #append to output list
        output_per_scenario.append(count)
        output_per_scenario.append(cmar[-1]-total_sh_mass)
        
        #populate the subhalo mass list
        sh_masses = masses[0:count]
        
        #determine the distribution of subhalos over the mass range
        mass_range=[10**6,10**7,10**8,10**9]
        mass_lower=[10**6,10**7,10**8]
        mass_upper=[10**7,10**8,10**9]
        for i in range(len(mass_lower)):
            indx=(sh_masses >= mass_lower[i])*(sh_masses<mass_upper[i])
            #append to the output list
            output_per_scenario.append(np.sum(indx))
            
        #generate random subhalo galactocentric radii following the mass profile of the logarithmic halo
        ran=np.random.rand(count)
        rad=np.linspace(r[0],r[-1],count)
        menc=[]
        for i in rad:
            menc.append(cp.mass(i/ro))
        menc=np.array(menc)
        sh_radii=np.interp(ran, menc/menc[-1], rad)
        
        #initialize the hernquist potentials for the subhalos
        sh_hern_r=[]
        for i in sh_masses:
            sh_hern_r.append(1.05*(i/(10**(8)))**(0.5))
        sh_pots=[]
        for i in range(len(sh_masses)):
            sh_pots.append(gp.HernquistPotential(sh_masses[i]/mo,sh_hern_r[i]/ro,ro=ro,vo=vo))
        
        #determine the circulr velocity of each subhalo at its galactocentric radius
        sh_circv=[]
        for i in sh_radii:
            sh_circv.append(tp.vcirc(i/ro))  
        sh_circv=np.array(sh_circv)
        
        #generate random initial conditions for subhalos
        shthea=np.arccos(1.-2.*np.random.rand(len(sh_radii)))
        shphi=2*np.pi*np.random.rand(len(sh_radii))
        shvR = (1/(np.sqrt(3)))*np.random.normal(0,sh_circv)
        shvT = (1/(np.sqrt(3)))*np.random.normal(0,sh_circv)
        shvZ = (1/(np.sqrt(3)))*np.random.normal(0,sh_circv)
        shR = sh_radii*np.sin(shthea)
        shZ = sh_radii*np.cos(shthea)
        
        #initialize and integrate the subhalo's orbits in the total potential
        shorb = np.column_stack([shR/ro,shvR/vo,shvT/vo,shZ/ro,shvZ/vo,shphi])
        sh_orbits = Orbit(shorb,ro=ro,vo=vo)
        sh_orbits.integrate(torb/to,tp)
        
        #create the moving potential
        moving_potential=[]
        for i in range(len(sh_orbits)):
            moving_potential.append(gp.MovingObjectPotential(sh_orbits[i],sh_pots[i]))
        moving_potential.append(sp)
        
        #integrate the tracer orbit in the moving potential
        tracer_orbit_sub = Orbit(tracerorb,ro=ro,vo=vo)
        tracer_orbit_sub.integrate(torb/to,moving_potential)
        
        #loop through the list of tracer orbits for information
        
        for i in range(0,len(tracer_r)):

            print('Started scenario '+str(ns+1)+' tracer '+str(i+1))
            
            #write output information for individual scenarios (items already appended to output_per_scenario: total sh count, analytic/dist mass difference, sh mass dist)
            output_per_scenario.append(tracer_orbit_sub[i].vx(torb[-1]/to))
            output_per_scenario.append(np.min(tracer_orbit_sub[i].vx(torb/to)))
            output_per_scenario.append(np.max(tracer_orbit_sub[i].vx(torb/to)))
            output_per_scenario.append(np.std(tracer_orbit_sub[i].vx(torb/to)))
            output_per_scenario.append(tracer_orbit_sub[i].vy(torb[-1]/to))
            output_per_scenario.append(np.min(tracer_orbit_sub[i].vy(torb/to)))
            output_per_scenario.append(np.max(tracer_orbit_sub[i].vy(torb/to)))
            output_per_scenario.append(np.std(tracer_orbit_sub[i].vy(torb/to)))
            output_per_scenario.append(tracer_orbit_sub[i].vz(torb[-1]/to))
            output_per_scenario.append(np.min(tracer_orbit_sub[i].vz(torb/to)))
            output_per_scenario.append(np.max(tracer_orbit_sub[i].vz(torb/to)))
            output_per_scenario.append(np.std(tracer_orbit_sub[i].vz(torb/to)))

            radius=tracer_orbit_sub[i].r(torb/to)
            
            output_per_scenario.append(tracer_orbit_sub[i].r(torb[-1]/to))
            output_per_scenario.append(np.min(radius))
            output_per_scenario.append(np.max(radius))
            output_per_scenario.append(np.std(radius))

            energy=tracer_orbit_sub[i].E(torb/to,pot=sp)

            output_per_scenario.append(tracer_orbit_sub[i].E(torb[-1]/to,pot=sp))
            output_per_scenario.append(np.min(energy))
            output_per_scenario.append(np.max(energy))
            output_per_scenario.append(np.std(energy))
                        
            energy_evolution_per_scenario[ns,i]=energy
            radius_evolution_per_scenario[ns,i]=radius

            #flush any errors when writing input info
            sys.stdout.flush()

        #flush any errors from main body loop
        sys.stdout.flush()
        
    #reshape the output list for easy parsing
    output_per_scenario=np.array(output_per_scenario)
    output_per_scenario=np.reshape(output_per_scenario,(num_scenarios,int(len(output_per_scenario)/num_scenarios)))

    #format the energy/radius variation array, take the variarion in each time slot per scenario. This is in style of Penarubbia 2019
    for i in range(0,len(tracer_r)):
        energy_evolution_per_tracer=energy_evolution_per_scenario[:,i,:]
        radius_evolution_per_tracer=radius_evolution_per_scenario[:,i,:]
        energy_evolution_sigma=np.std(energy_evolution_per_tracer,axis=0)
        radius_evolution_sigma=np.std(radius_evolution_per_tracer,axis=0)
        
        #compute stats for all scenario
        output_all_scenarios=[]
        totalsh_stat=[]
        massdif_stat=[]
        sub1_stat=[]
        sub2_stat=[]
        sub3_stat=[]
        vx_stat=[]
        vy_stat=[]
        vz_stat=[]
        r_stat=[]
        E_stat=[]
        for j in range(num_scenarios):
            totalsh_stat.append(output_per_scenario[j][0])
            massdif_stat.append(output_per_scenario[j][1])
            sub1_stat.append(output_per_scenario[j][2])
            sub2_stat.append(output_per_scenario[j][3])
            sub3_stat.append(output_per_scenario[j][4])
            vx_stat.append(output_per_scenario[j][5])
            vy_stat.append(output_per_scenario[j][9])
            vz_stat.append(output_per_scenario[j][13])
            r_stat.append(output_per_scenario[j][17])
            E_stat.append(output_per_scenario[j][21])

        #populate the output list for all scenario stats
        output_all_scenarios.append(np.mean(totalsh_stat))
        output_all_scenarios.append(np.std(totalsh_stat))
        output_all_scenarios.append(np.mean(massdif_stat))
        output_all_scenarios.append(np.std(massdif_stat))
        output_all_scenarios.append(np.mean(sub1_stat))
        output_all_scenarios.append(np.std(sub1_stat))
        output_all_scenarios.append(np.mean(sub2_stat))
        output_all_scenarios.append(np.std(sub2_stat))
        output_all_scenarios.append(np.mean(sub3_stat))
        output_all_scenarios.append(np.std(sub3_stat))
        output_all_scenarios.append(np.std(vx_stat))
        output_all_scenarios.append(np.std(vy_stat))
        output_all_scenarios.append(np.std(vz_stat))
        output_all_scenarios.append(np.mean(r_stat))
        output_all_scenarios.append(np.std(r_stat))
        output_all_scenarios.append(np.std(E_stat))
        output_all_scenarios.append(energy_evolution_sigma)

        #save file
        if sub_halo_mass==None:
            np.savetxt(str(tracer_r[i]) + 'kpctracer_' + str(num_scenarios)+'sc_'+str(smf)+'_smf_per',output_per_scenario)
            np.savetxt(str(tracer_r[i]) + 'kpctracer_' + str(num_scenarios)+'sc_'+str(smf)+'_smf_all',energy_evolution_sigma,header=str(output_all_scenarios[:-1]))
            np.savetxt(str(tracer_r[i]) + 'kpctracer_' + str(num_scenarios)+'sc_'+str(smf)+'_smf_rad',radius_evolution_sigma,header='Radius Evolution')
        elif sub_halo_mass!=None:
            np.savetxt(str(tracer_r[i]) + 'kpctracer_' + str(num_scenarios)+'sc_'+str(sub_halo_mass)+'shm_per',output_per_scenario)
            np.savetxt(str(tracer_r[i]) + 'kpctracer_' + str(num_scenarios)+'sc_'+str(sub_halo_mass)+'shm_all',energy_evolution_sigma,header=str(output_all_scenarios[:-1]))
            np.savetxt(str(tracer_r[i]) + 'kpctracer_' + str(num_scenarios)+'sc_'+str(sub_halo_mass)+'shm_rad',radius_evolution_sigma,header='Radius Evolution')

    #print the stats for all scenarios
    print(' ')
    print('Stats over all scenarios:')
    for i in range(len(all_info)):
        print(str(output_all_scenarios[i])+all_info[i])

    #print out c/d for comparison
    print('Mean subhalo hern-radius: '+str(np.mean(sh_hern_r)))
    print('Mean subhalo separation: ')

    #print a reference list for the per scenario info
    print(' ')
    print('A reference list for per scenario info (each [i-i+3] is of the form[final,min,max,std]):')
    print(['total # subs[0], analytic vs. dist mass dif[1], 10e6 subs[2], 10e7 subs[3], 10e8 subs[4], vx[5-8], vy[9-12], vz[13-16], r[17-20], E[21-24]'])
    print('Note that Energy Evolution can be accsessed easily by output_all_scenarios[-1].')
    
    return output_per_scenario, output_all_scenarios

#call function to begin simulations
per_scen, all_scen = substructure_integration(100,[5.,10.,20.,40.,60.])
