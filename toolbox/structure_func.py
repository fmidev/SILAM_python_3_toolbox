'''
Created on 5.4.2021

@author: sofievm
'''

import numpy as np
import numpy.f2py
from toolbox import gridtools, stations, drawer
from toolbox import supplementary as spp, silamfile
from toolbox import MyTimeVars as MTV
import os, shutil, glob
import datetime as dt
import netCDF4 as netcdf

#
# MPI may be tricky: in puhti it loads fine but does not work
# Therefore, first check for slurm loader environment. Use it if available
#
try:
    # slurm loader environment
    mpirank = int(os.getenv("SLURM_PROCID",None))
    mpisize = int(os.getenv("SLURM_NTASKS",None))
    chMPI = '_mpi%03g' % mpirank
    comm = None
    print('SLURM pseudo-MPI activated', chMPI, 'tasks: ', mpisize)
except:
    # not in pihti - try usual way
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        mpisize = comm.size
        mpirank = comm.Get_rank()
        chMPI = '_mpi%03g' % mpirank
        print ('MPI operation, mpisize=', mpisize, chMPI)
    except:
        print ("mpi4py failed, single-process operation")
        mpisize = 1
        mpirank = 0
        chMPI = ''
        comm = None

try:
    import struct_func_F as StF_F
    ifFortranOK = True
except:
    # Attention: sizes of the arrays must be at the end
    fortran_code_struct_func = '''
    
subroutine make_str_fun_map(mdlFld, patch_zero, ifNormalize, str_fun, iFilterRad, nxSTF, nySTF, nx, ny) 
  !
  ! Computes structure function for the field
  ! Note duality: if iFilterRad == 0, the whole map must be covered by the computations and 
  ! the array is (nx, ny, nx, ny)
  ! if iFilterRad > 0, only area around the cell +/- iFilterRad needs to be covered and the array
  ! is (nx, ny, 2*iFilterRad+1, 2*iFilterRad+1)
  !
  implicit none
  !
  ! Imported parameters
  real*4, dimension(0:nx-1, 0:ny-1), intent(in) :: mdlFld
  real*4, dimension(0:nx-1, 0:ny-1, 0:nxSTF-1, 0:nySTF-1), intent(out) :: str_fun
  integer, intent(in) :: patch_zero, iFilterRad
  logical, intent(in) :: ifNormalize
  ! sizes must be at the end
  integer*4, intent(in) :: nx, ny, nxSTF, nySTF

  ! Local variables
  integer :: ix, iy, ixFlt, iyFlt
  real*4 :: diff, fMean=0., fTmp

  ! If normalization is needed, have to make something to avoid div-zero
  if(ifNormalize)then
    fMean = sum(mdlFld) / (nx*ny)
  endif
  !
  ! For each grid cell, its squared difference from neighbours forms a filtering pattern
  ! 
  do iy = 0, ny-1
    do ix = 0, nx-1
      !
      ! iFilterRad can be given as zero or a positive value
      !
      if(iFilterRad == 0)then
        !
        ! Scan the whole grid
        !
        do iyFlt = 0, ny-1
          do ixFlt = 0, nx-1
            diff = mdlFld(ix, iy) - mdlFld(ixFlt, iyFlt)
            str_fun(ix,iy,ixFlt,iyFlt) = diff * diff
          end do
        end do
        ! Check what to do with the central grid cell
        select case(patch_zero)
          case(0)       ! leave zero
          case(1)       ! smoothen
! bad idea: other zeroes might exist, their importance must be higher than >0 elements
!            str_fun(ix,iy,ix,iy) = sum(str_fun(ix,iy,max(0,-1+ix):min(nx-1,1+ix), &
!                                                   & max(0,-1+iy):min(ny-1,1+iy))) / 8.0
            ! Find minimum non-zero element
            fTmp = str_fun(ix,iy,0,0)
            do iyFlt = 0, ny-1
              do ixFlt = 0, nx-1
                if(str_fun(ix,iy,ixFlt,iyFlt) > 0)then
                  if(fTmp > str_fun(ix,iy,ixFlt,iyFlt)) fTmp = str_fun(ix,iy,ixFlt,iyFlt)
                endif
              end do
            end do
            ! Replace zeroes with minimum value / 2, a random factor boosting their importance
            do iyFlt = 0, ny-1
              do ixFlt = 0, nx-1
                if(str_fun(ix,iy,ixFlt,iyFlt) == 0) str_fun(ix,iy,ixFlt,iyFlt) = fTmp / 2.0
              end do
            end do
          case(2)       ! prepare to nan
            str_fun(ix,iy,ix,iy) = -1  ! will later turn to nan
          case(3)       ! limit the range, taken later after all time steps are done
          case default
            print *,'patch_zero can be 0, 1, 2, 3, not:', patch_zero
            return
        end select
      else
        !
        ! Scan the neighbours within iFilterRad
        !
        do ixFlt = -iFilterRad, iFilterRad  ! max(-iFilterRad,-ix), min(iFilterRad-1, nx-ix)
          if(ix+ixFlt < 0 .or. ix + ixFlt >= nx)then
            str_fun(ix,iy,ixFlt+iFilterRad,:) = -1.
          else
            do iyFlt = -iFilterRad, iFilterRad  ! max(-iFilterRad,-iy), min(iFilterRad-1, ny-iy)
              if (iy + iyFlt < 0 .or. iy+iyFlt >= ny)then
                str_fun(ix,iy,ixFlt+iFilterRad,iyFlt+iFilterRad) = -1.
              else
                diff = mdlFld(ix, iy) - mdlFld(ix+ixFlt, iy+iyFlt)
                str_fun(ix,iy,ixFlt+iFilterRad,iyFlt+iFilterRad) = diff * diff
              endif   ! reasonable iy+iyFlt
            end do  ! iyFlt
          endif  ! reasonable ix+ixFlt 
        end do  !ixFlt
        ! Check what to do with the central grid cell
        select case(patch_zero)
          case(0)      ! do nothing
          case(1)      ! smoothen: take the smallest of the values around but not zero
            ! Find minimum non-zero element
            fTmp = str_fun(ix,iy,0,0)
            do iyFlt = 0, 2 * iFilterRad
              do ixFlt = 0, 2 * iFilterRad
                if(str_fun(ix,iy,ixFlt,iyFlt) > 0)then
                  if(fTmp > str_fun(ix,iy,ixFlt,iyFlt)) fTmp = str_fun(ix,iy,ixFlt,iyFlt)
                endif
              end do
            end do
            ! Replace zeroes with minimum value / 2, a random factor boosting their importance
            do iyFlt = 0, 2 * iFilterRad
              do ixFlt = 0, 2 * iFilterRad
                if(str_fun(ix,iy,ixFlt,iyFlt) == 0) str_fun(ix,iy,ixFlt,iyFlt) = fTmp / 2.0
              end do
            end do
          case(2)      ! prepare to nan
            str_fun(ix,iy,iFilterRad,iFilterRad) = -1  ! will later turn to nan
          case(3)       ! limit the range, taken later after all time steps are done
          case default
            print *,'patch_zero can be 0, 1, 2, 3, not:', patch_zero
            return
        end select  ! patch_zero
      endif  ! if iFilterRad given
      ! Normalize, if needed
      if(ifNormalize)then
        if(mdlFld(ix, iy) < 0.01 * fMean)then
          str_fun(ix,iy,:,:) = str_fun(ix, iy,:,:) / (fMean * fMean)
        else
          str_fun(ix,iy,:,:) = str_fun(ix,iy,:,:) / (mdlFld(ix, iy) * mdlFld(ix, iy))
        endif
      endif
    end do  ! ix
  end do  ! iy
end subroutine make_str_fun_map

!------------------------------------------------------------------------

subroutine make_str_fun_tsm(mdlFld, cells, patch_zero, ifNormalize, str_fun, & 
                          & iFilterRad, nxSTF, nySTF, nx, ny, nCells) 
  !
  ! Computes structure function for the given cells of the field
  ! Note duality: if iFilterRad == 0, the whole map must be covered by the computations and 
  ! the array is (nCells, nx, ny)
  ! if iFilterRad > 0, only area around the cell +/- iFilterRad needs to be covered and the array
  ! is (nCells, 2*iFilterRad+1, 2*iFilterRad+1)
  !
  implicit none
  !
  ! Imported parameters
  real*4, dimension(0:nx-1, 0:ny-1), intent(in) :: mdlFld
  real*4, dimension(0:nCells-1, 0:nxSTF-1, 0:nySTF-1), intent(out) :: str_fun
  integer*4, dimension(0:1, 0:nCells-1), intent(in) :: cells
  integer, intent(in) :: patch_zero, iFilterRad
  logical, intent(in) :: ifNormalize
  ! sizes must be at the end
  integer*4, intent(in) :: nx, ny, nxSTF, nySTF, nCells

  ! Local variables
  integer :: iCell, ix, iy, ixFlt, iyFlt
  real*4 :: diff, fMean=0., fTmp

  ! If normalization is needed, have to make something to avoid div-zero
  if(ifNormalize)then
    fMean = sum(mdlFld) / (nx*ny)
  endif
  !
  ! For each grid cell, its squared difference from neighbours forms a filtering pattern
  ! 
  do iCell = 0, nCells-1
    ix = cells(0, iCell)
    iy = cells(1, iCell)
    !
    ! iFilterRad can be zero (meaning, whole grid) or a positive value
    !
    if(iFilterRad == 0)then
      !
      ! Scan the whole grid
      !
      do iyFlt = 0, ny-1
        do ixFlt = 0, nx-1
          diff = mdlFld(ix, iy) - mdlFld(ixFlt, iyFlt)
          str_fun(iCell,ixFlt,iyFlt) = diff * diff
        end do
      end do
      ! Check what to do with the central grid cell
      select case(patch_zero)
        case(0)       ! leave zero
        case(1)       ! smoothen
! Bad idea: some neighbours appear more important than the centre point
!          str_fun(iCell,ix,iy) = sum(str_fun(iCell,max(0,-1+ix):min(nx-1,1+ix), &
!                                                 & max(0,-1+iy):min(ny-1,1+iy))) / 8.0
          ! Find minimum non-zero element
          fTmp = str_fun(iCell,0,0)
          do iyFlt = 0, ny-1
            do ixFlt = 0, nx-1
              if(str_fun(iCell,ixFlt,iyFlt) > 0)then
                if(fTmp > str_fun(iCell,ixFlt,iyFlt)) fTmp = str_fun(iCell,ixFlt,iyFlt)
              endif
            end do
          end do
          ! Replace _all_ zeroes with half of the the minimum value
          do iyFlt = 0, ny-1
            do ixFlt = 0, nx-1
              if(str_fun(iCell,ixFlt,iyFlt) == 0) str_fun(iCell,ixFlt,iyFlt) = fTmp / 2.0
            end do
          end do
        case(2)       ! prepare to nan
          str_fun(iCell,ix,iy) = -1  ! will later turn to nan
        case(3)       ! limit the range, taken later after all time steps are done
        case default
          print *,'patch_zero can be 0, 1, 2, 3, not:', patch_zero
          return
      end select
    else
      !
      ! Scan the neighbours
      !
      do ixFlt = -iFilterRad, iFilterRad  ! max(-iFilterRad,-ix), min(iFilterRad-1, nx-ix)
        if(ix+ixFlt < 0 .or. ix + ixFlt >= nx)then
          str_fun(iCell,ixFlt+iFilterRad,:) = -1.
        else
          do iyFlt = -iFilterRad, iFilterRad  ! max(-iFilterRad,-iy), min(iFilterRad-1, ny-iy)
            if (iy + iyFlt < 0 .or. iy+iyFlt >= ny)then
              str_fun(iCell,ixFlt+iFilterRad,iyFlt+iFilterRad) = -1.
            else
              diff = mdlFld(ix, iy) - mdlFld(ix+ixFlt, iy+iyFlt)
              str_fun(iCell,ixFlt+iFilterRad,iyFlt+iFilterRad) = diff * diff
            endif   ! reasonable iy+iyFlt
          end do  ! iyFlt
        endif  ! reasonable ix+ixFlt 
      end do  !ixFlt
      ! Check what to do with the central grid cell
      select case(patch_zero)
        case(0)      ! do nothing
        case(1)      ! smoothen
          str_fun(iCell,iFilterRad,iFilterRad) = sum(str_fun(iCell,max(0,-1+iFilterRad) : &
                                                                 & min(nx-1,1+iFilterRad), &
                                                                 & max(0,-1+iFilterRad) : &
                                                                 & min(ny-1,1+iFilterRad))) / 8.0
          ! Find minimum non-zero element
          fTmp = str_fun(iCell,0,0)
          do iyFlt = 0, 2 * iFilterRad
            do ixFlt = 0, 2 * iFilterRad
              if(str_fun(iCell,ixFlt,iyFlt) > 0)then
                if(fTmp > str_fun(iCell,ixFlt,iyFlt)) fTmp = str_fun(iCell,ixFlt,iyFlt)
              endif
            end do
          end do
          ! Replace zeroes with minimum value / 2, a random factor boosting their importance
          do iyFlt = 0, 2 * iFilterRad
            do ixFlt = 0, 2 * iFilterRad
              if(str_fun(iCell,ixFlt,iyFlt) == 0) str_fun(iCell,ixFlt,iyFlt) = fTmp / 2.0
            end do
          end do
        case(2)      ! prepare to nan
          str_fun(iCell,iFilterRad,iFilterRad) = -1  ! will later turn to nan
        case(3)       ! limit the range, taken later after all time steps are done
        case default
          print *,'patch_zero can be 0, 1, 2, 3, not:', patch_zero
          return
      end select
    endif   ! iFilterRad == 0
    ! Normalize, if needed
    if(ifNormalize)then
      if(mdlFld(ix, iy) < 0.01 * fMean)then
        str_fun(iCell,:,:) = str_fun(iCell,:,:) / (fMean * fMean)
      else
        str_fun(iCell,:,:) = str_fun(iCell,:,:) / (mdlFld(ix, iy) * mdlFld(ix, iy))
      endif
    endif
  end do  ! iCell
end subroutine make_str_fun_tsm

!----------------------------------------------------------------------------------

subroutine average_subdomains(mdlFld, arWinStart, arWinEnd, WinAvFld, nx, ny, nxSTF, nySTF)
  !
  ! Averages the model field over subdomins, whcih can overlap
  !
  implicit none
  
  ! Import parameters
  real*4, dimension(0:nx-1, 0:ny-1), intent(in) :: mdlFld
  integer*4, dimension(0:nxSTF-1, 0:nySTF-1, 0:1), intent(in) :: arWinStart, arWinEnd
  ! Output variable
  real*4, dimension(0:nxSTF-1, 0:nySTF-1), intent(out) :: WinAvFld
  ! Sizes
  integer, intent(in) :: nx, ny, nxSTF, nySTF

  ! Local variables
  integer :: ixSTF, iySTF
  
  ! Just go window by window taking average. Remember that go from 0 up to n-1 element
  do iySTF = 0, nySTF-1
    do ixSTF = 0, nxSTF-1
      WinAvFld(ixSTF, iySTF) = sum(mdlFld(arWinStart(ixSTF,iySTF,0):arWinEnd(ixSTF,iySTF,0)-1, &
                                        & arWinStart(ixSTF,iySTF,1):arWinEnd(ixSTF,iySTF,1)-1)) / &
                             & ((arWinStart(ixSTF,iySTF,0)-arWinEnd(ixSTF,iySTF,0)+1) * &
                              & (arWinStart(ixSTF,iySTF,1)-arWinEnd(ixSTF,iySTF,1)+1))
    end do  ! ixSTF
  end do  ! nySTF

end subroutine average_subdomains

'''

#    from numpy.distutils.fcompiler import new_fcompiler
#    compiler = new_fcompiler(compiler='intel')
#    compiler.dump_properties()

    # Compile the library and, if needed, copy it from the subdir where the compiler puts it
    # to the current directory
    #
    vCompiler = np.f2py.compile(fortran_code_struct_func,  modulename='struct_func_F', 
                                extra_args = '--opt=-O3', verbose=1, extension='.f90')
#                                extra_args = '--opt=-O3 --debug --f90flags=-fcheck=all', verbose=1, extension='.f90')
    if vCompiler == 0:
        cwd = os.getcwd()     # current working directory
        if os.path.exists(os.path.join('struct_func_F','.libs')):
            list_of_files = glob.glob(os.path.join('struct_func_F','.libs','*'))
            latest_file = max(list_of_files, key=os.path.getctime)
            shutil.copyfile(latest_file, os.path.join(cwd, os.path.split(latest_file)[1]))
        try: 
            import struct_func_F as StF_F
            ifFortranOK = True
        except:
            print('>>>>>>> FORTRAN failed-2, have to use Python. It will be SLO-O-O-O-O-O-OW')
            ifFortranOK = False
    else:
        print('>>>>>>> FORTRAN failed, have to use Python. It will be SLO-O-O-O-O-O-OW')
        ifFortranOK = False

#
# Conversion between string and integer for patch_zero switch
#
patch_zero_f = {'none': 0, 'smooth': 1, 'turn_to_nan' : 2, 'restrict_range' :3}

##############################################################################
#
# Class for structure function
#
##############################################################################

class structure_function():
    #
    # Manager for the structure function: creates various forms of StF, stores, reads
    # plots. etc.
    #
    def __init__(self, chTitle, gridSTF_factor, filterRad, tStart, nTimes, log):
        self.chTitle = chTitle
        self.filterRad = filterRad
        self.gridSTF_factor = gridSTF_factor
        self.structFunMap = None
        self.structFunTSM = None
        self.tStart = tStart
        self.StFtimes = [tStart]
        self.nTimes = nTimes
        self.log = log
        
    #==================================================================

    def make_StF_4_map(self, chMdlFNm, mdlVariable, patch_zero, ifNormalize):
        #
        # Computes the structure function using the model fields.
        # The function is computed for each gridSTF cell. Also, single out the 
        # station grid cells: those will be in the fitting procedure
        # patch_zero is the switch deciding the way the central grid cell is handled:
        # 0 = leave zero, 1 = patch with meadian, 2 = turn to nan
        #
        self.mdlVariable = mdlVariable
        fIn = silamfile.SilamNCFile(chMdlFNm)   # input file
        rdr = fIn.get_reader(mdlVariable)       # reader for the variable
        self.unit = '[(%s)^2]' % fIn._attrs[mdlVariable]['units']
        # store the model grid
        self.gridMdl = fIn.grid
        nxSTF = int(fIn.grid.nx / self.gridSTF_factor)
        nySTF = int(fIn.grid.ny / self.gridSTF_factor)
        self.gridSTF = gridtools.Grid(fIn.grid.x0, fIn.grid.dx * self.gridSTF_factor, nxSTF, 
                                      fIn.grid.y0, fIn.grid.dy * self.gridSTF_factor, nySTF,
                                      fIn.grid.proj)
        times = np.array(rdr.t())
        idx_tStart = np.searchsorted(times, self.tStart)
        if idx_tStart == 0 or idx_tStart== len(times):
            if not self.tStart in times:
                self.log.log(self.tStart.strftime('%Y-%m-%d %H:%M is not in times'))
                self.log.log('times are:' + '\n'.join(str(t) for t in times))
                raise ValueError
        #
        # structure function is a rectangular filter covering the essential
        # area around each grid cell.
        #
        if self.filterRad == 0:
            nxSTFd2 = self.gridSTF.nx
            nySTFd2 = self.gridSTF.ny
        else:
            nxSTFd2 = 2*self.filterRad+1
            nySTFd2 = nxSTFd2
        self.structFunMap = np.zeros(shape=(self.gridSTF.nx, self.gridSTF.ny, nxSTFd2, nySTFd2), 
                                     dtype=np.float32)
        iCnt = 0
        #
        # Structure function is a sum of squared differences
        #
#        rdr.goto(times[idx_tStart])  # set the reference time
        for iT in range(self.nTimes):
            print(idx_tStart + iT, times[idx_tStart +iT])
#            rdr.seek(min(iT,1))  # incremental to the last-read time
#            rdr.seek(idx_tStart + iT)  # much faster than goto(time)
            rdr.goto(times[idx_tStart +iT])
            try:
                mdlFld = rdr.read(1)
            except:
                self.log.log(times[idx_tStart +iT].strftime('Missing time: %Y-%m-%d %H:%M'))
                continue
            #
            # Reduce the resolution by the gridSTF_factor: average the nxn squares 
            #
            mdl4stf = np.mean(np.mean(mdlFld[:nxSTF*self.gridSTF_factor,
                                             :nySTF*self.gridSTF_factor,
                                             0].reshape((self.gridSTF.nx, self.gridSTF_factor,
                                                         self.gridSTF.ny, self.gridSTF_factor)),
                              axis=3),axis=1)
            # add the variances
            iCnt += 1
            if ifFortranOK:

                self.structFunMap += StF_F.make_str_fun_map(mdl4stf, patch_zero_f[patch_zero], 
                                                            ifNormalize, 
                                                            self.filterRad, nxSTFd2, nySTFd2, 
                                                            self.gridSTF.nx, self.gridSTF.ny)
                # handle negatives from FORTRAN: patch_zero may require nans
                if patch_zero == 'turn_to_nan': self.structFunMap[self.structFunMap < 0] = np.nan
            else:
                for ix in range(self.filterRad, self.gridSTF.nx-self.filterRad):
                    for iy in range(self.filterRad, self.gridSTF.ny-self.filterRad):
                        self.structFunMap[ix,iy,:,:] += (mdl4stf[ix-self.filterRad : ix+self.filterRad,
                                                                 iy-self.filterRad : iy+self.filterRad] -
                                                         mdl4stf[ix,iy])^2
                # handle the central points for the function: they are zeroes till here
                if patch_zero == 'none':     # leave
                    pass
                elif patch_zero == 'smooth':   # smoothen
                    self.structFunMap[ix,iy,
                                      self.filterRad+1,
                                      self.filterRad+1] = np.median(self.structFunMap[ix,iy,
                                                                                      self.filterRad:self.filterRad+2, 
                                                                                      self.filterRad:self.filterRad+2])
                elif patch_zero == turn_to_nan:   # exclude
                    self.structFunMap[ix,iy, self.filterRad+1, self.filterRad+1] = np.nan
        # turn to time average
        self.structFunMap /= iCnt
        #
        # Eliminate zeroes if needed
        if patch_zero != 'none': self.patch_zeroes_map(patch_zero)


    #==================================================================

    def make_StF_4_tsMatrix(self, chMdlFNm, mdlVariable, tsObs, patch_zero, ifNormalize):
        #
        # Generates the structure function for gridded observation locations
        # patch_zero is the switch deciding the way the central grid cell is handled:
        # 0 = leave zero, 1 = patch with meadian, 2 = turn to nan
        #
        self.mdlVariable = mdlVariable
        fIn = silamfile.SilamNCFile(chMdlFNm)   # input file
        rdr = fIn.get_reader(mdlVariable)       # reader for the variable
        self.unit = '[(%s)^2]' % fIn._attrs[mdlVariable]['units']
        times = np.array(rdr.t())
        idx_tStart = np.searchsorted(times, self.tStart)
        if idx_tStart == 0 or idx_tStart== len(times):
            if not self.tStart in times:
                self.log.log(self.tStart.strftime('%Y-%m-%d %H:%M is not in times'))
                self.log.log('times are:' + '\n'.join(str(t) for t in times))
                raise ValueError
        # store the model grid
        self.gridMdl = fIn.grid
        nxSTF = int(fIn.grid.nx / self.gridSTF_factor)
        nySTF = int(fIn.grid.ny / self.gridSTF_factor)
        self.gridSTF = gridtools.Grid(fIn.grid.x0, fIn.grid.dx * self.gridSTF_factor, nxSTF, 
                                      fIn.grid.y0, fIn.grid.dy * self.gridSTF_factor, nySTF,
                                      fIn.grid.proj)
        # project observations to the STF grid
        self.stations_StF = stations.downsample_to_grid(tsObs.stations, self.gridSTF)
        # indices of the cells with information
        cells = np.array(list(((int(s.code.split('_')[0]), int(s.code.split('_')[1])) for s in 
                               self.stations_StF))).T
        self.log.log('%s  fltr=%g, res_fct=%g, downsample stations from %g to %g' %
                     (self.chTitle, self.filterRad, self.gridSTF_factor, len(tsObs.stations),
                      cells.shape[1]))
        #
        # structure function is a rectangular filter covering either the essential
        # area around each station grid cell or the whole domain. Then, it is 3D 
        # (nStations, 2x rad, 2x rad) or (nStations, nxSTF, nySTF)
        #
        if self.filterRad == 0:
            nxSTFd2 = self.gridSTF.nx
            nySTFd2 = self.gridSTF.ny
        else:
            nxSTFd2 = 2*self.filterRad+1
            nySTFd2 = nxSTFd2

        self.structFunTSM = np.zeros(shape=(len(self.stations_StF), nxSTFd2, nySTFd2), 
                                     dtype=np.float32)
        iCnt = 0
        #
        # Structure function is a sum of squared differences
        #
#        rdr.goto(times[idx_tStart])  # set the reference time
        for iT in range(self.nTimes):
            print(idx_tStart + iT, times[idx_tStart +iT])
#            rdr.seek(min(iT,1))  # incremental to the last-read time
            rdr.goto(times[idx_tStart +iT])
            try:
                mdlFld = rdr.read(1)
            except:
                self.log.log(times[idx_tStart +iT].strftime('Missing time: %Y-%m-%d %H:%M'))
                continue
            #
            # Reduce the resolution by the gridSTF_factor: average the n x n squares 
            mdl4stf = np.mean(np.mean(mdlFld[:nxSTF*self.gridSTF_factor,
                                             :nySTF*self.gridSTF_factor,
                                             0].reshape((self.gridSTF.nx, self.gridSTF_factor,
                                                         self.gridSTF.ny, self.gridSTF_factor)),
                              axis=3),axis=1)
            # add the variances
            iCnt += 1
            if ifFortranOK:
#                print('Entering fortran')
                self.structFunTSM += StF_F.make_str_fun_tsm(mdl4stf, cells, patch_zero, ifNormalize, 
                                                            self.filterRad, nxSTFd2, nySTFd2,
                                                            self.gridSTF.nx, self.gridSTF.ny, 
                                                            cells.shape[1])
                # handle negatives from FORTRAN: patch_zero may require nans
                if patch_zero == 2: self.structFunTSM[self.structFunTSM < 0] = np.nan
#                print('Out of Fortran')
            else:                    
                for iSt in range(len(self.stations_StF)):
                    ix = cells[iSt][0]
                    iy = cells[iSt][1]
                    self.structFunTSM[iSt,:,:] += (mdl4stf[ix-self.filterRad:ix+self.filterRad,
                                                           iy-self.filterRad:iy+self.filterRad] -
                                                   mdl4stf[ix,iy])^2
        # Turn to time average
        self.structFunTSM /= iCnt
        # If needed, get rid of zeroes
        if patch_zero != 'none': self.patch_zero_TSM(patch_zero)


    #=================================================================
    
    def make_StF_4_subdomains_map(self, chMdlFNm, mdlVariable, chUnitForce, arWinStart,
                                  patch_zero, ifNormalize):
        #
        # Makes structure function for a bunch of squeared subdomains, whcih can overlap.
        # The subdomain size is given by the gridSTF_factor, their starting corners are in
        # arWinStart.
        # The output of the exercise is the StF for tsMatrix, where each subdomain is analogous 
        # to the station-cell.
        #
        self.mdlVariable = mdlVariable
        fIn = silamfile.SilamNCFile(chMdlFNm)   # input file
        rdr = fIn.get_reader(mdlVariable)       # reader for the variable
        if chUnitForce == '-': self.unit = '[(%s)^2]' % fIn._attrs[mdlVariable]['units']
        else: self.unit = chUnitForce
        times = np.array(rdr.t())
        idx_tStart = np.searchsorted(times, self.tStart)
        if idx_tStart == 0 or idx_tStart== len(times):
            if not self.tStart in times:
                self.log.log(self.tStart.strftime('%Y-%m-%d %H:%M is not in times'))
                self.log.log('times are:' + '\n'.join(str(t) for t in times))
                raise ValueError
        # store the model grid
        self.gridMdl = fIn.grid
        #
        # The StF grid is irregular in reality (subdomains overlap, possibly, irregularly)
        # but nxSTF and nySTF allow for faking a regular domain.
        #
        nxSTF, nySTF, iTmp = arWinStart.shape  # (nxSTF, nySTF, 2)
        dx_fake = self.gridMdl.dx * self.gridMdl.nx / nxSTF
        dy_fake = self.gridMdl.dy * self.gridMdl.ny / nySTF
        self.gridSTF = gridtools.Grid(fIn.grid.x0, dx_fake, nxSTF, 
                                      fIn.grid.y0, dy_fake, nySTF, fIn.grid.proj)
        #
        # structure function is a rectangular filter covering the essential
        # area around each station grid cell - or a full domain. It is 4D 
        # (nxSTF, nySTF, 2x rad, 2x rad) or (nxSTF, nySTF, nxSTF, nySTF), respectively
        #
        if self.filterRad == 0:
            nxSTFd2 = self.gridSTF.nx
            nySTFd2 = self.gridSTF.ny
        else:
            nxSTFd2 = 2*self.filterRad+1
            nySTFd2 = nxSTFd2

        self.structFunMap = np.zeros(shape=(nxSTFd2, nySTFd2, nxSTFd2, nySTFd2), dtype=np.float32)
        arWinEnd = arWinStart + self.gridSTF_factor
        iCnt = 0
        #
        # Structure function is a sum of squared differences
        #
#        rdr.goto(times[idx_tStart])  # set the reference time
        for iT in range(self.nTimes):
            print(idx_tStart + iT, times[idx_tStart +iT])
            rdr.goto(times[idx_tStart +iT])
#            rdr.seek(min(iT,1))  # incremental to the last-read time
            try:
                mdlFld_ = rdr.read(1)
            except:
                self.log.log(times[idx_tStart +iT].strftime('Missing time: %Y-%m-%d %H:%M'))
                continue
            try: mdlFld = mdlFld_.data
            except: mdlFld = mdlFld_
            #
            # Reduce the resolution by the gridSTF_factor: average the n x n squares 
            #
            # add the variances
            iCnt += 1
            if ifFortranOK:
#                print('Entering fortran')
                # Average the model field towards STF domains dimensions
                mdl4stf = StF_F.average_subdomains(mdlFld, arWinStart, arWinEnd, 
                                                   self.gridMdl.nx, self.gridMdl.ny, nxSTF, nySTF)
                # The structure function itself
                self.structFunMap += StF_F.make_str_fun_map(mdl4stf, patch_zero_f[patch_zero],
                                                            ifNormalize, 
                                                            self.filterRad, nxSTFd2, nySTFd2,
                                                            self.gridSTF.nx, self.gridSTF.ny)
                # handle negatives from FORTRAN: patch_zero may require nans
                if patch_zero == 2: self.structFunMap[self.structFunMap < 0] = np.nan
#                print('Out of Fortran')
            else:                    
                for ixSTF in range(nxSTF):
                    for iySTF in range(nySTF):
                        mdl4stf[ixSTF, iySTF] = np.ave(mdlFld[arWinStart[ixSTF,iySTF,0] :
                                                              arWinEnd[ixSTF,iySTF,0],
                                                              arWinStart[ixSTF,iySTF,1] : 
                                                              arWinEnd[ixSTF,iySTF,1], 0])
                for ix in range(nxSTF):
                    for iy in range(nySTF):
                        self.structFunMap[ix,iy,:,:] += (mdl4stf[ix-self.filterRad:ix+self.filterRad,
                                                                 iy-self.filterRad:iy+self.filterRad] -
                                                         mdl4stf[ix,iy])^2
        # Turn to time average
        self.structFunMap /= iCnt
        #
        # Eliminate zeroes if needed
        if patch_zero != 'none': self.patch_zeroes_map(patch_zero)


    #=================================================================
    
    def make_StF_4_subdomains_TSM(self, lstStations, chMdlFNm, mdlVariable, chUnitForce, arWinStart, 
                                  patch_zero, ifNormalize):
        #
        # Makes structure function for a bunch of squeared subdomains, whcih can overlap.
        # The subdomain size is given by the gridSTF_factor, their starting corners are in
        # arWinStart.
        # The output of the exercise is the StF for tsMatrix, where each subdomain is analogous 
        # to the station-cell.
        #
        self.mdlVariable = mdlVariable
        fIn = silamfile.SilamNCFile(chMdlFNm)   # input file
        rdr = fIn.get_reader(mdlVariable)       # reader for the variable
        if chUnitForce == '-': self.unit = '[(%s)^2]' % fIn._attrs[mdlVariable]['units']
        else: self.unit = chUnitForce
        times = np.array(rdr.t())
        idx_tStart = np.searchsorted(times, self.tStart)
        if idx_tStart == 0 or idx_tStart== len(times):
            if not self.tStart in times:
                self.log.log(self.tStart.strftime('%Y-%m-%d %H:%M is not in times'))
                self.log.log('times are:' + '\n'.join(str(t) for t in times))
                raise ValueError
        # store the model grid
        self.gridMdl = fIn.grid
        #
        # The StF grid is irregular in reality (subdomains overlap, possibly, irregularly)
        # but nxSTF and nySTF allow for faking a regular domain.
        #
        nxSTF, nySTF, iTmp = arWinStart.shape  # (nxSTF, nySTF, 2)
        dx_fake = self.gridMdl.dx * self.gridMdl.nx / nxSTF
        dy_fake = self.gridMdl.dy * self.gridMdl.ny / nySTF
        self.gridSTF = gridtools.Grid(fIn.grid.x0, dx_fake, nxSTF, 
                                      fIn.grid.y0, dy_fake, nySTF, fIn.grid.proj)
        #
        # Only the subdomains with stations are useful.
        #
        self.stations_StF = stations.downsample_to_subdomains(lstStations,
                                                              self.gridMdl,
                                                              arWinStart, 
                                                              self.gridSTF_factor)
        cells = np.array(list(((int(s.code.split('_')[0]), int(s.code.split('_')[1])) for s in 
                               self.stations_StF))).T
        self.log.log('%s  fltr=%g, res_fct=%g, downsample stations from %g to %g' %
                     (self.chTitle, self.filterRad, self.gridSTF_factor, len(lstStations), 
                      cells.shape[1]))
        #
        # structure function is a rectangular filter covering the essential
        # area around each station grid cell - or a full domain. It is 3D 
        # (nStations, 2x rad, 2x rad) or (nStations, nxSTF, nySTF), respectively
        #
        if self.filterRad == 0:
            nxSTFd2 = self.gridSTF.nx
            nySTFd2 = self.gridSTF.ny
        else:
            nxSTFd2 = 2*self.filterRad+1
            nySTFd2 = nxSTFd2

        self.structFunTSM = np.zeros(shape=(len(self.stations_StF), nxSTFd2, nySTFd2), 
                                     dtype=np.float32)
        arWinEnd = arWinStart + self.gridSTF_factor
        iCnt = 0
        #
        # Structure function is a sum of squared differences
        #
#        rdr.goto(times[idx_tStart])  # set the reference time
        for iT in range(self.nTimes):
            print(idx_tStart + iT, times[idx_tStart +iT])
            rdr.goto(times[idx_tStart +iT])
#            rdr.seek(min(iT,1))  # incremental to the last-read time
            try:
                mdlFld = rdr.read(1)
            except:
                self.log.log(times[idx_tStart +iT].strftime('Missing time: %Y-%m-%d %H:%M'))
                continue
            #
            # Reduce the resolution by the gridSTF_factor: average the n x n squares 
            #
            # add the variances
            iCnt += 1
            if ifFortranOK:
#                print('Entering fortran')
                # Average the model field towards STF domains dimensions
                mdl4stf = StF_F.average_subdomains(mdlFld, arWinStart, arWinEnd, 
                                                   self.gridMdl.nx, self.gridMdl.ny, nxSTF, nySTF)
                # The structure function itself
                self.structFunTSM += StF_F.make_str_fun_tsm(mdl4stf, cells, patch_zero_f[patch_zero],
                                                            ifNormalize, 
                                                            self.filterRad, nxSTFd2, nySTFd2,
                                                            self.gridSTF.nx, self.gridSTF.ny, 
                                                            cells.shape[1])
                # handle negatives from FORTRAN: patch_zero may require nans
                if patch_zero == 2: self.structFunTSM[self.structFunTSM < 0] = np.nan
#                print('Out of Fortran')
            else:                    
                for ixSTF in range(nxSTF):
                    for iySTF in range(nySTF):
                        mdl4stf[ixSTF, iySTF] = np.ave(mdlFld[arWinStart[ixSTF,iySTF,0] :
                                                              arWinEnd[ixSTF,iySTF,0],
                                                              arWinStart[ixSTF,iySTF,1] : 
                                                              arWinEnd[ixSTF,iySTF,1], 0])
                for iSt in range(len(self.stations_StF)):
                    ix = cells[iSt][0]
                    iy = cells[iSt][1]
                    self.structFunTSM[iSt,:,:] += (mdl4stf[ix-self.filterRad:ix+self.filterRad,
                                                           iy-self.filterRad:iy+self.filterRad] -
                                                   mdl4stf[ix,iy])^2
        # Turn to time average
        self.structFunTSM /= iCnt
        #
        # Eliminate zeroes if needed
        if patch_zero != 'none': self.patch_zeroes_TSM(patch_zero)

        
    #==================================================================
        
    def patch_zeroes_map(self, patch_zero):
        #
        # If time period is short, the structure function can accidentally be zero
        # That has to be removed if requested
        #
        if np.max(self.structFunMap) == 0:
            if patch_zero != 'none':
                self.log.log('All-zero structure function')
                return False
        elif patch_zero == 'from_neighbours':
            # zeroes are to be patched from neighbours
            nZeroes = np.sum(self.structFunMap == 0)
            nx = self.structFunMap.shape[2]
            ny = self.structFunMap.shape[3]
            while np.min(self.structFunMap) == 0:
                self.log.log('Removing %g zeroes in the structure function' % nZeroes)
                idxZero = np.argwhere(self.structFunMap == 0)
                for i0 in idxZero:
                    i,j,k,l = i0
                    self.structFunMap[i,j,k,l] = np.mean(self.structFunMap[i,j,
                                                                           max(0,k-1):min(k+1,nx),
                                                                           max(0,l-1):min(l+1,ny)])
                # Any progress?
                nZeroesNew = np.sum(self.structFunMap == 0)
                if nZeroes == nZeroesNew:
                    self.log.log('Cannot eliminate zeroes from Struct Function ' + self.chTitle)
                    raise ValueError('Cannot eliminate zeroes from Struct Function ' + self.chTitle)
                else:
                    nZeroes = nZeroesNew
        elif patch_zero == 'turn_to_nan':
            # zeroes are to be turned to nan
            self.structFunMap[self.structFunMap == 0] = np.nan
        elif patch_zero == 'restrict_range':
            # Limit the dynamic range of the function to 10000, i.e. square root range --> 100
            vMin = np.min(self.structFunMap[np.nonzero(self.structFunMap)])
            vMax = np.max(self.structFunMap)
            if vMin < 1e-15:
                self.log.log('Cannot limit dynamic range of Structure Function %s, min%g, max %g' %
                             (self.chTitle, vMin, vMax))
                vMin = 0.01 * vMax
                vCorr = 1.0
            else:
                vCorr = max(1.0, 0.01 * np.sqrt(vMax) / np.sqrt(vMin))
            # Even for correction =1, still need to do it to get rid of zeroes
            self.structFunMap = np.minimum(np.maximum(self.structFunMap, vMin * vCorr), vMax / vCorr)
        else:
            raise ValueError('Unknown patch_zero switch:' + str(patch_zero))
                
        
    #==================================================================
        
    def patch_zeroes_TSM(self, patch_zero):
        #
        # If time period is short, the structure function can accidentally be zero
        # That has to be removed if requested
        #
        if np.max(self.structFunTSM) == 0:
            if patch_zero > 0:
                self.log.log('All-zero structure function')
                raise ValueError
        elif patch_zero == 'from_neighbours':
            # zeroes are to be patched from neighbours
            nx = self.structFunTSM.shape[1]
            ny = self.structFunTSM.shape[2]
            nZeroes = np.sum(self.structFunTSM == 0)
            while np.min(self.structFunTSM) == 0:
                self.log.log('Removing %g zeroes in the structure function: ' % nZeroes)
                idxZero = np.argwhere(self.structFunTSM == 0)
                for i0 in idxZero:
                    i,k,l = i0
                    self.structFunTSM[i,k,l] = np.mean(self.structFunTSM[i,
                                                                         max(0,k-1):min(k+1,nx),
                                                                         max(0,l-1):min(l+1,ny)])
                # Any progress?
                nZeroesNew = np.sum(self.structFunTSM == 0)
                if nZeroesNew == nZeroes:
                    self.log.log('Cannot eliminate zeroes from Struct Function ' + self.chTitle)
                    raise ValueError
                else:
                    nZeroes = nZeroesNew
        elif patch_zero == 'turn_to_nan':
            # zeroes are to be turned to nan
            self.structFunTSM[self.structFunTSM == 0] = np.nan
        elif patch_zero == 'restrict_range':
            # Limit the dynamic range of the function to 10000, i.e. square root range --> 100
            vMin = np.min(self.structFunTSM[np.nonzero(self.structFunTSM)])
            vMax = np.max(self.structFunTSM)
            if vMin < 1e-15:
                self.log.log('Cannot limit dynamic range of Structure Function %s, min%g, max %g' %
                             (self.chTitle, vMin, vMax))
                vMin = 0.01 * vMax
                vCorr = 1.0
            else:
                vCorr = max(1.0, 0.01 * np.sqrt(vMax) / np.sqrt(vMin))
            if vCorr > 1.0:
                self.structFunTSM = np.minimum(np.maximum(self.structFunTSM, vMin * vCorr),
                                               vMax / vCorr)
        else:
            raise ValueError('Unknown patch_zero switch:' + str(patch_zero))


    #==================================================================

    def StF_to_netcdf(self, chOutFNm, ifClose):
        #
        # Writes the StFunction to nc4 file
        # The problem is that StF is not a standard SILAM file, so have to do all manually
        # Procedure follows the silamfile template
        # In particular, StF is 4-D map (nLon, nLat, nXSTF, nySTF)
        #
        # Open the file
        #
        self.log.log('Storing StFunction to file' + chOutFNm)
        try:
            f = netcdf.Dataset(chOutFNm, 'w', format="NETCDF4")
        except:
            self.log.log('>>>> Failed opening the nc file'  + chOutFNm)
            try:
                self.log.log('>>>> list of open files:' + '\n'.join(os.listdir('/proc/self/fd')))
            except:
                self.log.log('>>>> Cannot print the list of open files')
                self.log.log('>>>> Second try, see the exception')
                f = netcdf.Dataset(chOutFNm, 'w', format="NETCDF4")

        f.history = 'Writing StFunction: ' + self.chTitle
        
        # Two types of StF: for each grid cell of a map and for each station of tsMatrix
        # 
        if not self.structFunTSM is None:
            #
            # StF for stations, which is the main dimension. Note that lon and lat are variables
            # but not dimensions and their values are those of the station locations 
            #
            # First put stations to nc - that one creates lon and lat variables
            #
            stations.stations_to_nc(self.stations_StF, f)
            #
            # The StF dimensions can differ: either filter or whole domain
            #
            if self.filterRad == 0:
                # StFunction covers the whole domain for all central locations
                f.createDimension('lonSTF', self.gridSTF.nx)
                f.createDimension('latSTF', self.gridSTF.ny)
                lonSTF = f.createVariable('lonSTF', 'f', ('lonSTF',))
                latSTF = f.createVariable('latSTF', 'f', ('latSTF',))
                lonSTF[:] = self.gridSTF.x()
                latSTF[:] = self.gridSTF.y()
            else:
                # StF covers neighborhood around the central point, i.e. is 
                # geographically centred around each central point
                f.createDimension('lonSTF', 2 * self.filterRad + 1)
                f.createDimension('latSTF', 2 * self.filterRad + 1)
                lonSTF = f.createVariable('lonSTF', 'f', ('lonSTF',))
                latSTF = f.createVariable('latSTF', 'f', ('latSTF',))
                lonSTF[:] = range(2*self.filterRad+1)
                latSTF[:] = range(2*self.filterRad+1)

        elif not self.structFunMap is None:
            #
            # The map-based StF: lon and lat are the main dimensions
            #
            f.createDimension('lon', self.gridSTF.nx)
            lon = f.createVariable('lon', 'f', ('lon',))
            lon[:] = self.gridSTF.x()
            f.createDimension('lat', self.gridSTF.ny)
            lat = f.createVariable('lat', 'f', ('lat',))
            lat[:] = self.gridSTF.y()
            # StF dimensions can differ: either filter or whole domain
            if self.filterRad == 0:
                # StFunction covers the whole domain for all central locations
                dimSTF = ''
                lonSTF = f.createVariable('lonSTF', 'f', ('lon',))
                latSTF = f.createVariable('latSTF', 'f', ('lat',))
                lonSTF[:] = self.gridSTF.x()
                latSTF[:] = self.gridSTF.y()
            else:
                # StF covers neighborhood around the central point, i.e. is 
                # geographically centred around each central point
                dimSTF = 'STF'
                f.createDimension('lonSTF', 2 * self.filterRad + 1)
                f.createDimension('latSTF', 2 * self.filterRad + 1)
                lonSTF = f.createVariable('lonSTF', 'f', ('lonSTF',))
                latSTF = f.createVariable('latSTF', 'f', ('latSTF',))
                lonSTF[:] = range(2*self.filterRad+1)
                latSTF[:] = range(2*self.filterRad+1)
        else:
            self.log.log('Neight Map nor tsMatrix StF defined')
            raise ValueError

        f.variables['lon'].units = 'degrees_east'
        f.variables['lat'].units = 'degrees_north'
        lonSTF.units = 'degrees_east'
        latSTF.units = 'degrees_north'

        # Time, for e.g. seasonal StF
        f.createDimension('time', None)   #len(arrTime))  # to make it unlimited record dimension
        time = f.createVariable('time', 'i', ('time',))
        time[:] = [ (h - self.tStart).total_seconds() for h in self.StFtimes]  #    arrTime
        time.calendar = 'standard'
        time.units = self.tStart.strftime('seconds since %Y-%m-%d %H:%M:%S UTC')
        time.long_name = 'time'
        time.standard_name = 'time'

        varNm = 'StF_' + self.mdlVariable
        #
        # File opened. Write the StFunction
        #
        if not self.structFunTSM is None:
            vSTF = f.createVariable(varNm, 'f', ('time','station','lonSTF','latSTF'), 
                                    zlib=True, complevel=5, 
                                    chunksizes=(1,1,self.gridSTF.nx, self.gridSTF.ny),
                                    least_significant_digit=3, fill_value=-999)
            vSTF[0,:,:,:] = self.structFunTSM[:,:,:]
            
        elif not self.structFunMap is None:
            # Beware of lon,lat order! Opposite to SILAM netcdf
            vSTF = f.createVariable(varNm, 'f', ('time','lon','lat','lon'+dimSTF,'lat'+dimSTF), 
                                    zlib=True, complevel=5, 
                                    chunksizes=(1,1,1,self.gridSTF.nx,self.gridSTF.ny),
                                    least_significant_digit=3, fill_value=-999)
            vSTF[0,:,:,:,:] = self.structFunMap[:,:,:,:]
        f.variables[varNm].units = self.unit
        #
        # Job done but do not rush t olose the file: more times may show up. StF 
        # can be big, writing step by step is an option
        #
        if ifClose: f.close()


    #==================================================================

    def StF_from_netcdf(self, chInFNm):
        #
        # Reads the StFunction from netcdf file
        #
        # open netcdf file
        with netcdf.Dataset(chInFNm) as ncIn: 
            # stations
            self.stations_StF = self.stations_from_nc(ncIn)
            # function itself
            self.structFunTSM = ncIn.variables['StF_' + self.chSpeciesNm][0,:,:,:]
        return self


    #===================================================================
    
    def draw(self, chOutDir, chOutFNmTempl, fraction2draw):
        #
        # Draws teh structure function as a series of maps
        #
        if not self.structFunMap is None:
            if self.filterRad == 0:
                # The StF covers the whole domain. A multitude of pictures 
                for ix in range(0, self.gridSTF.nx, int(1. / np.sqrt(fraction2draw))):
                    for iy in range(0, self.gridSTF.ny, int(1. / np.sqrt(fraction2draw))):
                        arTmp = self.structFunMap[ix, iy,:,:]
                        drawer.draw_map_with_points('%s %03g_%03g Rad=%g, res_fct=%g, %s, %g times' % 
                                                    (self.chTitle, ix, iy, self.filterRad, 
                                                     self.gridSTF_factor, self.unit, self.nTimes),
                                                    self.gridSTF.x(), self.gridSTF.y(), self.gridSTF, 
                                                    chOutDir, chOutFNmTempl  + 
                                                    '_map_cell_%03g_%03g' % (ix, iy), 
                                                    vals_points=None, 
                                                    vals_map = arTmp[:,::-1],
                                                    chUnit='') #, numrange=(1e-2,100))
            else:
                # The StF covers the vicinity of each point a single multi-square picture
                arTmp = np.zeros(shape=(self.gridSTF.nx, self.gridSTF.ny), dtype=np.float32)
                for ix in range(filterRad+1, self.gridSTF.nx, filterRad*2+2):
                    for iy in range(filterRad+1, self.gridSTF.ny, filterRad*2+2):
                        xmin = max(ix-filterRad, 0)
                        xmin_sf = xmin - ix + filterRad
                        ymin = max(iy-filterRad, 0)
                        ymin_sf = ymin - iy + filterRad
                        xmax = min(ix+filterRad, self.gridSTF.nx-1)
                        xmax_sf = filterRad - (ix - xmax)
                        ymax = min(iy+filterRad, self.gridSTF.ny-1)
                        ymax_sf = filterRad - (iy - ymax)
            #            print('xmin, xmax, ymin, ymax', xmin, xmax, ymin, ymax)
            #            print('xmin_sf, xmax_sf, ymin_sf, ymax_sf', xmin_sf, xmax_sf, ymin_sf, ymax_sf)
                        arTmp[xmin:xmax+1,ymin:ymax+1] += self.structFunMap[ix,iy,
                                                                           xmin_sf : xmax_sf+1, 
                                                                           ymin_sf : ymax_sf+1]
                drawer.draw_map_with_points('%s %03g_%03g Rad=%g, res_fct=%g, %s, %g times' % 
                                            (self.chTitle, ix, iy, self.filterRad, 
                                             elf.gridSTF_factor, self.unit, self.nTimes),
                                            self.gridSTF.x(), self.gridSTF.y(), 
                                            self.gridSTF, 
                                            chOutDir, chOutFNmTempl + 
                                            '_map_cell_%03g_%03g' % (ix, iy),
                                            vals_points=None, 
                                            vals_map = arTmp[:,::-1],
                                            chUnit='') #, numrange=(1e-2,100))
            
        elif not self.structFunTSM is None:
                    # Station-based StF. A multitude of pictures
                    for iSt, st in enumerate(StF.stations_StF):
                        if iSt % int(1. / fraction2draw) != 0: continue
                        if filterRad == 0:
                            arTmp = StF.structFunTSM[iSt,:,:]
                        else:
                            arTmp = np.zeros(shape=(StF.gridSTF.nx, StF.gridSTF.ny), dtype=np.float32)
                            fx, fy = StF.gridSTF.geo_to_grid(st.lon, st.lat)
                            ix = np.round(fx).astype(np.int32)
                            iy = round(fy).astype(np.int32)
                            xmin = max(ix-filterRad, 0)
                            xmin_sf = xmin - ix + filterRad
                            ymin = max(iy-filterRad, 0)
                            ymin_sf = ymin - iy + filterRad
                            xmax = min(ix+filterRad, StF.gridSTF.nx-1)
                            xmax_sf = filterRad - (ix - xmax)
                            ymax = min(iy+filterRad, StF.gridSTF.ny-1)
                            ymax_sf = filterRad - (iy - ymax)
                #            print('xmin, xmax, ymin, ymax', xmin, xmax, ymin, ymax)
                #            print('xmin_sf, xmax_sf, ymin_sf, ymax_sf', xmin_sf, xmax_sf, ymin_sf, ymax_sf)
                            arTmp[xmin:xmax+1,ymin:ymax+1] = StF.structFunTSM[iSt, xmin_sf : xmax_sf+1,
                                                                              ymin_sf : ymax_sf+1]
                        drawer.draw_map_with_points('%s %03g_%03g Rad=%g, res_fct=%g, %s, %g times' %
                                                    (self.chTitle, ix, iy, self.filterRad, 
                                                     elf.gridSTF_factor, self.unit, self.nTimes),
                                                    StF.gridSTF.x(), StF.gridSTF.y(), StF.gridSTF, 
                                                    chOutDir, chOutFNmTempl + '_TSM_stat_%03g' % iSt,
                                                    vals_points=None, 
                                                    vals_map = arTmp[:,::-1],
                                                    chUnit='') #, numrange=(1e-2,100))
        else:
            self.log.log('Neither map nor tsMatrix-based struture function exists')



#####################################################################################
#####################################################################################

if __name__ == '__main__':
    
    chDirMain = 'd:\\project\\COPERNICUS\\CAMS_63_Ensemble\\FUSE_v1_0'
#    chMdlFNm = 'd:\project\COPERNICUS\CAMS_63_Ensemble\data\SILAM\\SILAM_%y4%m2%d2_NO2_0H24H.nc'
    chMdlFNm = 'f:\\project\\CAMS63_ensemble\\AQ_2018\\CAMS_2018\\SILAM\\SILAM_%y4%m2%d2_NO2_0H24H.nc'
    fusion_window_hrs = 240
    patch_zero = 2           # 0 = do nothing, 1 = smoothen, 2 = nan 
    
    ifTestStructFuncMap = True
    ifTestStructFunctsMatrix = True

    #------------------------------------------------------------------------------
    if ifTestStructFuncMap:
        chOutDir = os.path.join(chDirMain,'struct_func_map','pics_tstSTF_%ghrs'  % fusion_window_hrs)
        # Test structure function for maps
        iProcess = 0
        for tStart in list( (dt.datetime(2018,iMon+1,1) for iMon in range(12))):
            for filterRad in [0]:  #20,40,60]:
                for gridSTF_factor in [20, 15, 10, 7, 5]: #, 4, 3, 2, 1]:
                    # Parallelise
                    iProcess += 1
                    if np.mod(iProcess-1, mpisize) != mpirank: continue
                    spp.ensure_directory_MPI(os.path.join(chOutDir, 'fct_%02g' % gridSTF_factor))
                    #
                    # Define the structure function object
                    #
                    StF = structure_func.structure_function(
                        'tst_StF_map', gridSTF_factor, filterRad, tStart, fusion_window_hrs, 
                        spp.log(os.path.join(chOutDir, 'struct_fun_map_rad_%g_fct_%g_%s.log' % 
                                             (filterRad, gridSTF_factor, tStart.strftime('%Y%m%d')))))
                    # The structure function
                    #
                    StF.make_StF_full_map(chMdlFNm, 'NO2', patch_zero, ifNormalize=False)

                    StF.StF_to_netcdf(os.path.join(chOutDir,'struct_fun_map_rad_%g_fct_%g_%s.nc4' % 
                                                   (filterRad, gridSTF_factor, 
                                                    tStart.strftime('%Y%m%d'))), ifClose=True)
                    continue
                    # Draw the structure function for a few locations
                    #
                    if filterRad == 0:
                        for ix in range(0, StF.gridSTF.nx, int(StF.gridSTF.nx/10)):
                            for iy in range(0, StF.gridSTF.ny, 10):
                                arTmp = StF.structFunMap[ix, iy,:,:]
                                drawer.draw_map_with_points('NO2_StF %03g_%03g Rad=%g, res_fct=%g, %s, %s - %s' % 
                                                            (ix, iy, filterRad, gridSTF_factor, StF.unit, 
                                                             tStart.strftime('%Y-%m-%d %H'),
                                                             (tStart + 
                                                              spp.one_hour*fusion_window_hrs).strftime('%m-%d %H')),
                                                            StF.gridSTF.x(), StF.gridSTF.y(),
                                                            StF.gridSTF, 
                                                            os.path.join(chOutDir, 'fct_%02g' % 
                                                                         gridSTF_factor), 
                                                            'NO2_StFmap_%03g_%03g_fltr_%g_fct_%02g_%s' % 
                                                            (ix, iy, filterRad, gridSTF_factor,
                                                             tStart.strftime('%Y%m%d')), 
                                                            vals_points=None, 
                                                            vals_map = arTmp[:,::-1],
                                                            chUnit='') #, numrange=(1e-2,100))
                    else:
                        arTmp = np.zeros(shape=(StF.gridSTF.nx, StF.gridSTF.ny), dtype=np.float32)
                        for ix in range(filterRad+1, StF.gridSTF.nx, filterRad*2+2):
                            for iy in range(filterRad+1, StF.gridSTF.ny, filterRad*2+2):
                                xmin = max(ix-filterRad, 0)
                                xmin_sf = xmin - ix + filterRad
                                ymin = max(iy-filterRad, 0)
                                ymin_sf = ymin - iy + filterRad
                                xmax = min(ix+filterRad, StF.gridSTF.nx-1)
                                xmax_sf = filterRad - (ix - xmax)
                                ymax = min(iy+filterRad, StF.gridSTF.ny-1)
                                ymax_sf = filterRad - (iy - ymax)
                    #            print('xmin, xmax, ymin, ymax', xmin, xmax, ymin, ymax)
                    #            print('xmin_sf, xmax_sf, ymin_sf, ymax_sf', xmin_sf, xmax_sf, ymin_sf, ymax_sf)
                                arTmp[xmin:xmax+1,ymin:ymax+1] += StF.structFunMap[ix,iy,
                                                                                   xmin_sf : xmax_sf+1, 
                                                                                   ymin_sf : ymax_sf+1]
                        drawer.draw_map_with_points('NO2_StF Rad=%g, res_fct=%g, %s, %s - %s' % 
                                                    (filterRad, gridSTF_factor, StF.unit,
                                                     tStart.strftime('%Y-%m-%d:%H'),
                                                     (tStart + 
                                                      spp.one_hour*fusion_window_hrs).strftime('%m-%d %H')),
                                                    StF.gridSTF.x(), StF.gridSTF.y(), 
                                                    StF.gridSTF, 
                                                    chOutDir, 
                                                    'NO2_structfun_fltr_%g_fct_%02g_%s' % 
                                                    (filterRad, gridSTF_factor, 
                                                     tStart.strftime('%Y%m%d')), 
                                                    vals_points=None, 
                                                    vals_map = arTmp[:,::-1],
                                                    chUnit='') #, numrange=(1e-2,100))
                    StF.log.close()

    #------------------------------------------------------------------------------
    if ifTestStructFunctsMatrix:
        #
        # Test structure function for tsMatrices
        #
        tsObs = MTV.TsMatrix.fromNC('d:\\data\\measurements\\EEA2018\\obs_NO2.nc')
        chOutDir = os.path.join(chDirMain,'struct_func_TSM\\pics_tstSTF_%ghrs'  % fusion_window_hrs)
        
        iProcess = 0
        for tStart in list( (dt.datetime(2018,iMon+1,1) for iMon in range(12))):
            for filterRad in [0]: #20,40,60]:
                for gridSTF_factor in [20, 15, 10, 7, 5, 4, 3, 2, 1]:
                    # Parallelise
                    iProcess += 1
                    if np.mod(iProcess-1, mpisize) != mpirank: continue
                    spp.ensure_directory_MPI(os.path.join(chOutDir, 'fct_%02g' % gridSTF_factor))

                    StF = structure_func.structure_function(
                        'tst_StF_tsM', gridSTF_factor, filterRad, tStart, fusion_window_hrs, 
                        spp.log(os.path.join(chOutDir, 'struct_fun_tsMatr_rad_%g_fct_%g_%s.log' % 
                                             (filterRad, gridSTF_factor, tStart.strftime('%Y%m%d')))))
                    # The structure function
                    #
                    StF.make_StF_4_tsMatrix(chMdlFNm, 'NO2', tsObs, patch_zero, ifNormalize=False)
                
                    StF.StF_to_netcdf(os.path.join(chOutDir,'struct_fun_tsMatr_rad_%g_fct_%g_%s.nc4' % 
                                                   (filterRad, gridSTF_factor, 
                                                    tStart.strftime('%Y%m%d'))), ifClose=True)
                    continue
                    # Draw the structure function for all downsampled combo-stations
                    # The structure function for each station will be plotted on the main map 
                    #
                    # Not more than 100 pictures:
                    iSkip = max(1, np.round(len(StF.stations_StF) / 100.0).astype(np.int16))
                    # Draw
                    for iSt, st in enumerate(StF.stations_StF):
                        if iSt % iSkip != 0: continue
                        if filterRad == 0:
                            arTmp = StF.structFunTSM[iSt,:,:]
                        else:
                            arTmp = np.zeros(shape=(StF.gridSTF.nx, StF.gridSTF.ny), dtype=np.float32)
                            fx, fy = StF.gridSTF.geo_to_grid(st.lon, st.lat)
                            ix = np.round(fx).astype(np.int32)
                            iy = round(fy).astype(np.int32)
                            xmin = max(ix-filterRad, 0)
                            xmin_sf = xmin - ix + filterRad
                            ymin = max(iy-filterRad, 0)
                            ymin_sf = ymin - iy + filterRad
                            xmax = min(ix+filterRad, StF.gridSTF.nx-1)
                            xmax_sf = filterRad - (ix - xmax)
                            ymax = min(iy+filterRad, StF.gridSTF.ny-1)
                            ymax_sf = filterRad - (iy - ymax)
                #            print('xmin, xmax, ymin, ymax', xmin, xmax, ymin, ymax)
                #            print('xmin_sf, xmax_sf, ymin_sf, ymax_sf', xmin_sf, xmax_sf, ymin_sf, ymax_sf)
                            arTmp[xmin:xmax+1,ymin:ymax+1] = StF.structFunTSM[iSt, xmin_sf : xmax_sf+1, 
                                                                              ymin_sf : ymax_sf+1]
                        drawer.draw_map_with_points('NO2_StF %s Rad=%g, res_fct=%g, %s %s - %s' % 
                                                    (st.code, filterRad, gridSTF_factor, StF.unit,
                                                     tStart.strftime('%Y-%m-%d %H'),
                                                     (tStart + 
                                                      spp.one_hour*fusion_window_hrs).strftime('%m-%d %H')),
                                                    StF.gridSTF.x(), StF.gridSTF.y(), StF.gridSTF, 
                                                    os.path.join(chOutDir, 'fct_%02g' % gridSTF_factor), 
                                                    'NO2_structfun_cell_%s_fltr_%g_fct_%02g_%s' % 
                                                    (st.code, filterRad, gridSTF_factor, 
                                                     tStart.strftime('%Y%m%d')), 
                                                    vals_points=None, 
                                                    vals_map = arTmp[:,::-1],
                                                    chUnit='') #, numrange=(1e-2,100))
                    StF.log.close()
