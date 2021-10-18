import os, sys
import numpy as np
import datetime as dt
from toolbox import gradsfile as gf
#from support import pupynere as netcdf
from sys import argv
import netCDF4 as netcdf

print ('Hello :)')

def open_ncF_out(fNm, arrLon, arrLat, arrLev, anTime, arrTime, vars3d, vars2d, units, hst=None):        
    print('opening nc file ' + fNm)
    f = netcdf.Dataset(fNm, 'w')
    if hst: f.history = hst

    ncVars = {}

    nx = len(arrLon)
    ny = len(arrLat)
    nz = len(arrLev)
    nt = len(arrTime)

    f.createDimension('lon', nx)
    lon = f.createVariable('lon', 'f', ('lon',))
    lon[:] = arrLon
    lon.units = 'degrees_east'

    f.createDimension('lat', ny)
    lat = f.createVariable('lat', 'f', ('lat',))
    lat[:] = arrLat
    lat.units = 'degrees_north'

    f.createDimension('height', nz)
    level = f.createVariable('height', 'f', ('height',))
    level[:] = arrLev
    level.units = 'height_from_surface'

    f.createDimension('time', nt)
    time = f.createVariable('time', 'i', ('time',))
    time[:] = [ (h - anTime).total_seconds()/3600 for h in arrTime ]  #    arrTime
    print(anTime, type(anTime))
    time.units = anTime.strftime('hours since %Y-%m-%d %H:%M:%S')

    for varNm in vars3d:
        f.createVariable(varNm, 'f', ('time', 'height', 'lat', 'lon'))
        f.variables[varNm].units = units[varNm]    

    for varNm in vars2d:
        f.createVariable(varNm, 'f', ('time', 'lat', 'lon'))
        f.variables[varNm].units = units[varNm]
    
    return f

if __name__ == '__main__':
    case = argv[1]
    mnth = int(argv[2])
    sDy = int(argv[3])
    nDays = int(argv[4])
    
    acid_file = '/arch/silam/work/project/TRANSPHORM/TRANSFORM_acid_'+ case + '/TRANSFORM_acid_20051231.grads.ctl'
    cb4_file = '/arch/silam/work/project/TRANSPHORM/TRANSFORM_cb4_'+ case + '/TRANSFORM_cb4_20051231.grads.ctl'
    fire_file = '/arch/silam/work/project/TRANSPHORM/TRANSFORM_fireFix_2005_s53/TRANSFORM_fire_20051231.grads.ctl'
    dust_file = '/arch/silam/work/project/TRANSPHORM/TRANSFORM_dustFix_2005/TRANSFORM_20051231.grads.ctl'  
    
    
    outDir = '/arch/silam/work/project/TRANSPHORM/TRANSFORM_nc_dustFix_'+case
    if not os.path.exists(outDir): os.mkdir(outDir)
    nc_out_file = os.path.join(outDir,'SILAM_'+case)
    
    cb4Sps = ['O3', 'HCHO', 'CO']
    fireSps = ['FPM2_5', 'FPM_10']
    dustSps = ['DUST2_5', 'DUST10']
    
    species = {
               'SO2':    ['SO2_gas*64.0e6'], 
               'SO4':    ['SO4_m_70*96.0e6',
                          'SO4_m_20*96.0e6',
                          'NH415SO4_f*96.0e6',
                          'NH415SO4_c*96.0e6'], 
               'NO':     ['NO_gas*30.0e6'], 
               'NO2':    ['NO2_gas*46.0e6'], 
               'NO32_5': ['NH4NO3_m_70*62.0e6',
                          'NO3_C_m3_0*16.74e6'],       #0.27 in pm2.5 (from emep)
               'NO3':    ['NH4NO3_m_70*62.0e6',
                          'NO3_C_m3_0*62.0e6'], 
               'HNO3':   ['HNO3_gas*63.0e6'], 
               'O3':     ['O3_gas*48.0e6'], 
               'NH3':    ['NH3_gas*17.0e6'], 
               'HCHO':   ['HCHO_gas*30.0e6'], 
               'CO':     ['CO_gas*28.0e6'], 
               'NH4':    ['NH4NO3_m_70*18.0e6',
                          'NH415SO4_f*18.0e6',
                          'NH415SO4_c*18.0e6'], 
               'BaP':    ['BAP_m_70*1.0e9'], 
               'EC':     ['BC_m_70*1.0e9'],
               'PPM2_5': ['BC_m_70*1.0e9',
                          'NONBC_m_70*1.0e9'], 
               'PPM10':  ['BC_m_70*1.0e9',
                          'NONBC_m_70*1.0e9', 
                          'PM_m6_0*1.0e9'], 
               'FPM2_5': ['PM_FRP_m1_5*1.0e9'],
               'FPM_10': ['PM_FRP_m1_5*1.0e9', 
                          'PM_FRP_m6_0*1.0e9'],
               'SSLT2_5':['SSLT_m_05*1.0e9', 
                          'SSLT_m_50*1.0e9', ], 
               'SSLT10': ['SSLT_m_05*1.0e9', 
                          'SSLT_m_50*1.0e9', 
                          'SSLT_m3_0*1.0e9'],
               'DUST2_5':['DUST_m1_5*1.0e9'],
               'DUST10': ['DUST_m1_5*1.0e9', 
                          'DUST_m6_0*1.0e9'],
               'PM2_5':  ['SO4_m_70*98.0e6',
                          'SO4_m_20*98.0e6',
                          'NH415SO4_f*123.0e6',
                          'NH415SO4_c*123.0e6',
                          'NH4NO3_m_70*80.0e6',
                          'NO3_C_m3_0*7.155e6',   # 0.27 in pm2.5 (from emep) * (NO3 molar mass - chlorine that escaped from seasalt)
                          'BC_m_70*1.0e9',
                          'NONBC_m_70*1.0e9', 
                          'SSLT_m_05*1.0e9', 
                          'SSLT_m_50*1.0e9', 
                          #'DUST_m1_5*1.0e9',
                          #'PM_FRP_m1_5*1.0e9'
                          ], 
               'PM10':   ['SO4_m_70*98.0e6',
                          'SO4_m_20*98.0e6',
                          'NH415SO4_f*123.0e6',
                          'NH415SO4_c*123.0e6',
                          'NH4NO3_m_70*80.0e6',
                          'NO3_C_m3_0*26.5e6',
                          'BC_m_70*1.0e9',
                          'NONBC_m_70*1.0e9', 
                          'PM_m6_0*1.0e9',
                          'SSLT_m_05*1.0e9', 
                          'SSLT_m_50*1.0e9', 
                          'SSLT_m3_0*1.0e9',
                          #'DUST_m1_5*1.0e9',
                          #'DUST_m6_0*1.0e9',
                          #'PM_FRP_m1_5*1.0e9',
                          #'PM_FRP_m6_0*1.0e9'
                          ]}
    
    cnc = {}
    wd = {}
    dd = {}
    v3D = []
    v2D = []
    units = {}
    for sp in species:
        if sp in cb4Sps:
            f = cb4_file
        elif sp in fireSps:
            f = fire_file
        elif sp in dustSps:
            f = dust_file
        else:
            f = acid_file
        print(sp, f)
        gv = 'cnc_'+'+cnc_'.join(species[sp])
        cnc[sp] = gf.Gradsfile(f, gv)
        v3D.append('cnc_'+sp)
        gv = 'dd_'+'+dd_'.join(species[sp])
        dd[sp] = gf.Gradsfile(f, gv)
        gv = 'wd_'+'+wd_'.join(species[sp])
        wd[sp] = gf.Gradsfile(f, gv)
         
        v3D.append('cnc_'+sp)
        v2D.append('dd_'+sp)
        v2D.append('wd_'+sp)
        v2D.append('col_'+sp)
        units['cnc_'+sp]='ug m-3'
        units['dd_'+sp]='ug m-2 h-1'
        units['wd_'+sp]='ug m-2 h-1'
        units['col_'+sp]='ug m-2'
    
    
    u = gf.Gradsfile(acid_file, 'disp_u_wind') 
    units['u']='m s-1'
    v3D.append('u')
    v = gf.Gradsfile(acid_file, 'disp_v_wind') 
    units['v']='m s-1'
    v3D.append('v')
    BLH = gf.Gradsfile(acid_file, 'BLH')
    units['BLH']='m'
    v2D.append('BLH')
    prec_rate = gf.Gradsfile(acid_file, 'prec_rate*3.6e3')
    units['prec_rate']='mm h-1'
    v2D.append('prec_rate')
    cloud_cover = gf.Gradsfile(acid_file, 'Total_cloud')
    units['cloud_cover']='fraction'
    v2D.append('cloud_cover')
    T2m = gf.Gradsfile(acid_file, 'temp_2m')
    units['T2m']='K'
    v2D.append('T2m')
    
    arrX = T2m.x()
    arrY = T2m.y()
    fZ = u.z()
    
    dz = [fZ[0]*2.0]
    for iz in range(1,len(fZ)):
        dz.append((fZ[iz]-np.sum(dz[:iz]))*2.0)
    
    arrZ = []
    inZ = [0, 2, 4, 5, 7]
    for iz in range(len(inZ)):    
        arrZ.append(fZ[inZ[iz]])
        
    
    #now = dt.datetime(2005,1,1,0,0,0)
    now = dt.datetime(2005,mnth,sDy,0,0,0)
    
    #nDays = 31
    #nDays = 1
    for dy in range(nDays):
        print(now.strftime('%Y %m %d'))
        julDay = now - dt.datetime(2005,1,1,0,0,0)
        arrT = np.array([julDay.days*24 + hr for hr in range(24)])
        file_out = open_ncF_out(nc_out_file+now.strftime('%Y%m%d')+'.nc', arrX, arrY, arrZ, 
                                dt.datetime(2005,1,1,0,0,0), arrT, v3D, v2D, units, 
                                hst='FMI SILAM model output')
    
        for hr in range(24):
            
            print(hr)
            for sp in cnc:
                cnc[sp].goto(now)
                fld = cnc[sp].read() 
                if sp == 'PM2_5':
                    cnc['FPM2_5'].goto(now)
                    fld = fld + cnc['FPM2_5'].read()
                    cnc['DUST2_5'].goto(now)
                    fld = fld + cnc['DUST2_5'].read()
                if sp == 'PM10':
                    cnc['FPM_10'].goto(now)
                    fld = fld + cnc['FPM_10'].read()  
                    cnc['DUST10'].goto(now)
                    fld = fld + cnc['DUST10'].read()  
                for iz in range(len(arrZ)):
                    file_out.variables['cnc_'+sp][hr,iz,:,:] = fld[:,:,inZ[iz]].T
                ave = np.average(fld,axis=2,weights=dz,returned=True)
                fld = ave[0]*ave[1]
                file_out.variables['col_'+sp][hr,:,:] = fld.T
            for sp in dd:
                dd[sp].goto(now)
                fld = dd[sp].read() 
                if sp == 'PM2_5':
                    dd['FPM2_5'].goto(now)
                    fld = fld + dd['FPM2_5'].read()
                    dd['DUST2_5'].goto(now)
                    fld = fld + dd['DUST2_5'].read()
                if sp == 'PM10':
                    dd['FPM_10'].goto(now)
                    fld = fld + dd['FPM_10'].read()  
                    dd['DUST10'].goto(now)
                    fld = fld + dd['DUST10'].read()                 
                file_out.variables['dd_'+sp][hr,:,:] = fld.T * 3600.0        
            for sp in wd:
                wd[sp].goto(now)
                fld = wd[sp].read() 
                if sp == 'PM2_5':
                    wd['FPM2_5'].goto(now)
                    fld = fld + wd['FPM2_5'].read()
                    wd['DUST2_5'].goto(now)
                    fld = fld + wd['DUST2_5'].read()
                if sp == 'PM10':
                    wd['FPM_10'].goto(now)
                    fld = fld + wd['FPM_10'].read()                
                    wd['DUST10'].goto(now)
                    fld = fld + wd['DUST10'].read()                
                file_out.variables['wd_'+sp][hr,:,:] = fld.T * 3600.0
    
            u.goto(now)
            fld = u.read() 
            for iz in range(len(arrZ)):
                file_out.variables['u'][hr,iz,:,:] = fld[:,:,inZ[iz]].T        
            v.goto(now)
            fld = v.read() 
            for iz in range(len(arrZ)):
                file_out.variables['v'][hr,iz,:,:] = fld[:,:,inZ[iz]].T
            BLH.goto(now)
            fld = BLH.read() 
            file_out.variables['BLH'][hr,:,:] = fld.T 
            prec_rate.goto(now)
            fld = prec_rate.read() 
            file_out.variables['prec_rate'][hr,:,:] = fld.T
            cloud_cover.goto(now)
            fld = cloud_cover.read() 
            file_out.variables['cloud_cover'][hr,:,:] = fld.T
            T2m.goto(now)
            fld = T2m.read() 
            file_out.variables['T2m'][hr,:,:] = fld.T    
        
        
            now = now + dt.timedelta(hours = 1)
        
        
        file_out.close()
    
    print('Bye :)')
