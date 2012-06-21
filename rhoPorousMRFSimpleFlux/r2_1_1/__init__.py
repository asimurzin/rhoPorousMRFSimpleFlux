#!/usr/bin/env python

#--------------------------------------------------------------------------------------
## pythonFlu - Python wrapping for OpenFOAM C++ API
## Copyright (C) 2010- Alexey Petrov
## Copyright (C) 2009-2010 Pebble Bed Modular Reactor (Pty) Limited (PBMR)
## 
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.
## 
## See http://sourceforge.net/projects/pythonflu
##
## Author : Alexey PETROV, Andrey SIMURZIN
##


#---------------------------------------------------------------------------
from Foam import ref, man


#---------------------------------------------------------------------------
def _createFields( runTime, mesh, simple ):
    ref.ext_Info() << "Reading thermophysical properties\n" << ref.nl
    
    thermo = man.basicPsiThermo.New( mesh )

    rho = man.volScalarField( man.IOobject( ref.word( "rho" ),
                                            ref.fileName( runTime.timeName() ),
                                            mesh,
                                            ref.IOobject.READ_IF_PRESENT,
                                            ref.IOobject.AUTO_WRITE ),
                              man.volScalarField( thermo.rho(), man.Deps( thermo ) ) )

    p = man.volScalarField( thermo.p(), man.Deps( thermo ) )
    h = man.volScalarField( thermo.h(), man.Deps( thermo ) )
    psi = man.volScalarField( thermo.psi(), man.Deps( thermo ) )
   
    ref.ext_Info() << "Reading field U\n" << ref.nl
    U = man.volVectorField( man.IOobject( ref.word( "U" ),
                                          ref.fileName( runTime.timeName() ),
                                          mesh,
                                          ref.IOobject.MUST_READ,
                                          ref.IOobject.AUTO_WRITE ),
                            mesh )

    phi = man.compressibleCreatePhi( runTime, mesh, rho, U )
    
    pRefCell = 0
    pRefValue = 0.0
    
    pRefCell, pRefValue = ref.setRefCell( p, simple.dict(), pRefCell, pRefValue )
    
    rhoMax = ref.dimensionedScalar( simple.dict().lookup( ref.word( "rhoMax" ) ) )
    rhoMin = ref.dimensionedScalar( simple.dict().lookup( ref.word( "rhoMin" ) ) )
    
    ref.ext_Info() << "Creating turbulence model\n" << ref.nl
    turbulence = man.compressible.RASModel.New( rho,
                                                U,
                                                phi,
                                                thermo )
    
    initialMass = ref.fvc.domainIntegrate( rho )
    
    return thermo, rho, p, h, psi, U, phi, pRefCell, pRefValue, turbulence, initialMass, rhoMax, rhoMin


#--------------------------------------------------------------------------------------
def createZones( mesh, U, simple ):
    mrfZones = man.MRFZones( mesh )
    mrfZones.correctBoundaryVelocity( U )

    pZones = man.thermalPorousZones( mesh )
    pressureImplicitPorosity = ref.Switch( False )
    
    # nUCorrectors used for pressureImplicitPorosity
    nUCorr = 0
    if pZones.size():
        # nUCorrectors for pressureImplicitPorosity
        tmp, nUCorr = simple.dict().readIfPresent( ref.word( "nUCorrectors" ), nUCorr )
        if nUCorr > 0:
            pressureImplicitPorosity = ref.Switch( True )
            ref.ext_Info()<< "Using pressure implicit porosity" << ref.nl
            pass
        else:
            ref.ext_Info() << "Using pressure explicit porosity" << ref.nl
            pass

    return mrfZones, pZones, pressureImplicitPorosity, nUCorr


#--------------------------------------------------------------------------------------
def _UEqn( phi, U, p, rho, turbulence, mrfZones, pZones, pressureImplicitPorosity, nUCorr ):
    # Solve the Momentum equation
    UEqn = man.fvVectorMatrix( turbulence.divDevRhoReff( U ), man.Deps( turbulence ) )  + man.fvm.div( phi, U )
    
    UEqn.relax()
    
    mrfZones.addCoriolis( rho, UEqn )
    
    trAU = None
    trTU = None

    if pressureImplicitPorosity:
        tTU = man.volTensorField( ref.tensor( ref.I ) * UEqn.A(), man.Deps( UEqn ) )
        pZones.addResistance( UEqn, tTU )
        trTU = man.volTensorField( tTU.inv(), man.Deps( tTU ) )
        trTU.rename( ref.word( "rAU" ) )

        gradp = ref.fvc.grad(p)

        for UCorr in range( nUCorr ):
            U << ( trTU() & ( UEqn.H() - gradp ) ) # mixed calculations
            pass
        U.correctBoundaryConditions()
        pass
    else:
        pZones.addResistance( UEqn )
        ref.solve( UEqn == -ref.fvc.grad( p ) )
        trAU = man.volScalarField( 1.0 / UEqn.A(), man.Deps( UEqn ) )
        trAU.rename( ref.word( "rAU" ) )
        pass
    
    return UEqn, trAU, trTU


#--------------------------------------------------------------------------------------
def _hEqn( U, phi, h, turbulence, rho, p, thermo, pZones ):
    
    hEqn = ref.fvm.div( phi, h ) - ref.fvm.Sp( ref.fvc.div( phi ), h ) - ref.fvm.laplacian( turbulence.alphaEff(), h ) \
           == - ref.fvc.div( phi, 0.5 * U.magSqr(), ref.word( "div(phi,K)" ) )

    pZones.addEnthalpySource( thermo, rho, hEqn )
    
    hEqn.relax()
    hEqn.solve()

    thermo.correct()
    pass


#--------------------------------------------------------------------------------------
def _pEqn( runTime,mesh, UEqn, rho, thermo, psi, U, p, phi, trTU, trAU, mrfZones, \
           pRefCell, pRefValue, pressureImplicitPorosity, cumulativeContErr, simple, rhoMin, rhoMax  ):
    if pressureImplicitPorosity:
        U << ( trTU() & UEqn.H() ) # mixed calculations
        pass
    else:
        U << ( trAU() * UEqn.H() )  # mixed calculations
        pass
    
    # UEqn.clear()
    
    closedVolume = False
    
    if simple.transonic(): 
       phid = surfaceScalarField( ref.word( "phid" ), 
                                  ref.fvc.interpolate( psi ) * ( ref.fvc.interpolate( U ) & mesh.Sf() ) )

       mrfZones.relativeFlux( ref.fvc.interpolate( psi ), phid )
   
       while simple.correctNonOrthogonal():
           if pressureImplicitPorosity:
               tpEqn = ref.fvc.div( phid, p ) - ref.fvm.laplacian( rho * trTU, p )
               pass
           else:
               tpEqn = ref.fvc.div( phid, p ) - ref.fvm.laplacian( rho * trAU, p )
               pass

           tpEqn.setReference( pRefCell, pRefValue )
           tpEqn.solve()

           if simple.finalNonOrthogonalIter():
               phi == tpEqn.flux()
               pass
           pass
    else:
        phi << ( ref.fvc.interpolate( rho * U ) & mesh.Sf() )
        mrfZones.relativeFlux( ref.fvc.interpolate( rho ), phi )
        
        closedVolume = ref.adjustPhi( phi, U, p )

        while simple.correctNonOrthogonal():
            if pressureImplicitPorosity:
                tpEqn = ( ref.fvm.laplacian( rho * trTU, p ) == ref.fvc.div( phi ) )
                pass
            else:
                tpEqn = ( ref.fvm.laplacian( rho * trAU, p ) == ref.fvc.div( phi ) )
                pass
            tpEqn.setReference( pRefCell, pRefValue )

            tpEqn.solve()

            if simple.finalNonOrthogonalIter():
                phi -= tpEqn.flux()
                pass
            pass
            

    cumulativeContErr = ref.ContinuityErrs( phi, runTime, mesh, cumulativeContErr )
    
    # Explicitly relax pressure for momentum corrector
    p.relax()
    
    if pressureImplicitPorosity:
        U -= ( trTU() & ref.fvc.grad(p) ) # mixed calculations
        pass
    else:
        U -= trAU() * ref.fvc.grad(p) # mixed calculations
        pass
    
    U.correctBoundaryConditions()
    
    # For closed-volume cases adjust the pressure and density levels
    # to obey overall mass continuity
    if closedVolume:
        p += ( initialMass - ref.fvc.domainIntegrate( psi * p ) ) / ref.fvc.domainIntegrate( psi )
        pass
    
    rho << thermo.rho()
    rho << rho.ext_max( rhoMin )
    rho << rho.ext_min( rhoMax )
    rho.relax()
    ref.ext_Info() << "rho max/min : " << rho.ext_max().value() << " " << rho.ext_min().value() << ref.nl
    
    pass
    
    return cumulativeContErr


#--------------------------------------------------------------------------------------
def main_standalone( argc, argv ):

    args = ref.setRootCase( argc, argv )

    runTime = man.createTime( args )

    mesh = man.createMesh( runTime )
    
    simple = man.simpleControl( mesh )

    thermo, rho, p, h, psi, U, phi, pRefCell, pRefValue, turbulence, initialMass, rhoMax, rhoMin = _createFields( runTime, mesh, simple )
    
    mrfZones, pZones, pressureImplicitPorosity, nUCorr = createZones( mesh, U, simple )
    
    cumulativeContErr = ref.initContinuityErrs()

    ref.ext_Info() << "\nStarting time loop\n" << ref.nl
    
    while simple.loop() :
        ref.ext_Info() << "Time = " << runTime.timeName() << ref.nl << ref.nl
                
        # Pressure-velocity SIMPLE corrector
        UEqn, trAU, trTU = _UEqn( phi, U, p, rho, turbulence, mrfZones, pZones, pressureImplicitPorosity, nUCorr )
            
        _hEqn( U, phi, h, turbulence, rho, p, thermo, pZones )
        
        cumulativeContErr = _pEqn( runTime,mesh, UEqn, rho, thermo, psi, U, p, phi, trTU, trAU, mrfZones, \
                                                             pRefCell, pRefValue, pressureImplicitPorosity, cumulativeContErr, simple, rhoMin, rhoMax  )
        
        turbulence.correct()
        
        runTime.write()

        ref.ext_Info() << "ExecutionTime = " << runTime.elapsedCpuTime() << " s" << \
              "  ClockTime = " << runTime.elapsedClockTime() << " s" << ref.nl << ref.nl
        
        pass

    ref.ext_Info() << "End\n"

    import os
    return os.EX_OK


#--------------------------------------------------------------------------------------
import sys, os
from Foam import FOAM_VERSION
if FOAM_VERSION( ">=", "020101" ):
   if __name__ == "__main__" :
      argv = sys.argv
      os._exit( main_standalone( len( argv ), argv ) )
      pass
   pass   
else:
   from Foam.OpenFOAM import ext_Info
   ext_Info()<< "\nTo use this solver, It is necessary to SWIG OpenFoam2.1.1 or higher \n "


    
#--------------------------------------------------------------------------------------
