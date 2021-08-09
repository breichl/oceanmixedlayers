# This is a Python import of the matlab code to compute the mixed layer depth (MLD) of an input profile.
# This is based on the MATLAB code maintained on mixedlayer.ucsd.edu

import numpy as np
from .threshold import threshold as threshold
from .gradient import gradient as gradient
from .extrema import extrema as extrema

class holtetalley():

    def algorithm_mld(presi, sali, cnsrv_tempi, ptntl_rhoi):

        pres=np.copy(presi)
        sal = np.copy(sali)
        cnsrv_temp=np.copy(cnsrv_tempi)
        ptntl_rho=np.copy(ptntl_rhoi)

        if ((len(pres.shape))==1 and (len(cnsrv_temp.shape))==1):
            pres = np.atleast_2d(pres).T
            cnsrv_temp = np.atleast_2d(cnsrv_temp).T
            sal = np.atleast_2d(sal).T
            ptntl_rho = np.atleast_2d(ptntl_rho).T
        elif ((len(pres.shape))==1 and (len(cnsrv_temp.shape))>1):
            pres = (np.broadcast_to(pres,np.shape(cnsrv_temp.T)).T).copy()

        LP = np.shape(pres)[0]
        LS = np.shape(pres)[1:]

        MINDIFF = np.nanmin((pres-10.)**2,axis=0)
        for zi in range(LP):
            LI = (MINDIFF==(pres[zi,...]-10.)**2)
            pres[:LP-zi,LI]=(pres[zi:,LI])
            pres[LP-zi:,LI]=np.NaN
            sal[:LP-zi,LI]=(sal[zi:,LI])
            sal[LP-zi:,LI]=np.NaN
            ptntl_rho[:LP-zi,LI]=(ptntl_rho[zi:,LI])
            ptntl_rho[LP-zi:,LI]=np.NaN
            cnsrv_temp[:LP-zi,LI]=(cnsrv_temp[zi:,LI])
            cnsrv_temp[LP-zi:,LI]=np.NaN
            MINDIFF[LI]=0.0

        ##########################################################################
        # Calculate the MLD using a threshold method with de Boyer Montegut et al's
        # criteria; a density difference of .03 kg/m^3 or a temperature difference
        # of .2 degrees C.  The measurement closest to 10 dbar is used as the
        # reference value.  The threshold MLDs are interpolated to exactly match
        # the threshold criteria.

        mldepthdens,mldepthdens_index = threshold.threshold_mld_fixedcoord(pres, ptntl_rho, 0.03, 0.0, interpsurf=False,
                                                                           interp=True, absdiff=True)
        # The threshold looks for exceedance, so we use the negative temperature.
        mldepthptmp,mldepthptmp_index = threshold.threshold_mld_fixedcoord(pres, -cnsrv_temp, 0.2, 0.0, interpsurf=False,
                                                                           interp=True, absdiff=True)

        # # The gradient MLD values
        ddmin, ddmin_index = gradient.gradient_mld_fixedcoord(pres, -ptntl_rho,0.0005,smooth=False)
        dsmin, dsmin_index = gradient.gradient_mld_fixedcoord(pres, -sal,1.e+10,smooth=True)
        dtmax, dtmax_index = gradient.gradient_mld_fixedcoord(pres, cnsrv_temp,0.005,smooth=False)
        dtdzmax, dtdzmax_index = gradient.gradient_mld_fixedcoord(pres, cnsrv_temp,1.e10,smooth=True)
        dddzmax, dddzmax_index = gradient.gradient_mld_fixedcoord(pres, -ptntl_rho,1.e10,smooth=True)

        # # The maxima or minima of the temperature, salinity, and potential density
        # # profiles
        tmax,tmax_index = extrema.maxval_mld_fixedcoord(pres,cnsrv_temp)
        smin,smin_index = extrema.maxval_mld_fixedcoord(pres,-sal)
        dmin,dmin_index = extrema.maxval_mld_fixedcoord(pres,-ptntl_rho)

        # Fit method MLDs
        upperdtmin, upperdtmin_index = gradient.linearfit_mld_fixedcoord(pres, cnsrv_temp, 1.e-10,smooth=True)
        upperdsmax, upperdsmax_index = gradient.linearfit_mld_fixedcoord(pres, -sal, 1.e-10,smooth=True)
        upperddmax, upperddmax_index = gradient.linearfit_mld_fixedcoord(pres, -ptntl_rho, 1.e-10,smooth=True)

        # # Sometimes subsurface temperature or salinity intrusions exist at the base
        # # of the mixed layer.  For temperature, these intrusions are
        # # characterized by subsurface temperature maxima located near temperature
        # # gradient maxima. If the two maxima are separated by less than deltad,
        # # the possible MLD value is recorded in dtandtmax.
        dtandtmax = np.zeros(np.shape(mldepthdens))
        dtandtmax_index = np.zeros(np.shape(mldepthdens),dtype=int)
        dtmax2,dtmax2_index = gradient.max_gradient(pres, cnsrv_temp,1.e8,smooth=True)

        if_seplt = abs(dtmax2-tmax) < 100
        dtandtmax[if_seplt] = np.minimum(dtmax2[if_seplt], tmax[if_seplt])
        dtandtmax_index[if_seplt] = np.minimum(dtmax2_index[if_seplt], tmax_index[if_seplt])

        dsandsmin = np.zeros(np.shape(mldepthdens))
        dsandsmin_index = np.zeros(np.shape(mldepthdens),dtype=int)
        dsmin2,dsmin2_index = gradient.max_gradient(pres, -sal,1.e+10,smooth=True)
        if_seplt = abs(dsmin2-smin) < 100
        dsandsmin[if_seplt] = np.minimum(dsmin2[if_seplt], smin[if_seplt])
        dsandsmin_index[if_seplt] = np.minimum(dsmin2_index[if_seplt], smin_index[if_seplt])

        # #########################################################################

        # # To determine if the profile resembles a typical winter or summer profile,
        # # the temperature change across the thermocline, tdiff, is calculated and
        # # compared to the temperature cutoff. tdiff is calculated as the
        # # temperature change between the intersection of the mixed layer and thermocline fits and a
        # # point two depth indexes deeper.  If upperdtmin is set to 0 or at the
        # # bottom of the profile, the points from the thermocline fit are used
        # # to evaluate tdiff.

        if_incolumn = (upperdtmin_index>0) & (upperdtmin_index<LP-2)
        # if  upperdtmin(mldindex)>0 && upperdtmin(mldindex)<(m-2)
        m = len(upperdtmin_index[if_incolumn])
        tdiff = np.zeros(np.shape(upperdtmin_index))
        tdiff[if_incolumn] = (cnsrv_temp[:,if_incolumn][upperdtmin_index[if_incolumn], range(m)] -
                              cnsrv_temp[:,if_incolumn][upperdtmin_index[if_incolumn]+2, range(m)])
        #     tdiff(mldindex) = temp(dtdzmax(mldindex)-1)-temp(dtdzmax(mldindex)+1);

        m2 = len(upperdtmin_index[~if_incolumn])
        tdiff[~if_incolumn] = (cnsrv_temp[:,~if_incolumn][dtdzmax_index[~if_incolumn]-1, range(m2)] -
                               cnsrv_temp[:,~if_incolumn][dtdzmax_index[~if_incolumn]+1, range(m2)])

        # # tdiff is compared to the temperature cutoffs
        tcutoffu = .5;   # Upper temperature cutoff, used to initially classify profiles as winter or summer profiles -- degrees C
        tcutoffl = -.25; # Lower temperature cutoff, used to initially classify profiles as winter or summer profiles -- degrees C
        testt = (tdiff > tcutoffl) & (tdiff<tcutoffu)

        # # For salinity and potential density profiles, the potential density
        # # change across the pycnocline is calculated in a similar manner and
        # # compared to a potential density cutoff.
        if_incolumn = (upperddmax_index>0) & (upperddmax_index<LP-2)
        m = len(upperddmax_index[if_incolumn])
        m2 = len(upperddmax_index[~if_incolumn])
        ddiff = np.zeros(np.shape(upperddmax_index))
        
        ddiff[if_incolumn] = (ptntl_rho[:,if_incolumn][upperddmax_index[if_incolumn], range(m)] -
                              ptntl_rho[:,if_incolumn][upperddmax_index[if_incolumn]+2, range(m)])
        #     tdiff(mldindex) = temp(dtdzmax(mldindex)-1)-temp(dtdzmax(mldindex)+1);
        ddiff[~if_incolumn] = (ptntl_rho[:,~if_incolumn][dddzmax_index[~if_incolumn]-1, range(m2)] -
                               ptntl_rho[:,~if_incolumn][dddzmax_index[~if_incolumn]+1, range(m2)])
        testd = np.copy(testt)
        dcutoff = -.06;      # Density cutoff, used to initially classify profiles as winter or summer profiles -- kg/m^3
        testd[(ddiff > dcutoff) & (tdiff>tcutoffu)] = True
        testd[(ddiff > dcutoff) & (tdiff<tcutoffl)] = False

        rangee=25
        mixedt = np.atleast_1d(np.zeros(LS))
        analysis_t = np.atleast_1d(np.zeros(LS))
        mixeds = np.atleast_1d(np.zeros(LS))
        analysis_s = np.atleast_1d(np.zeros(LS))
        mixedd = np.atleast_1d(np.zeros(LS))
        analysis_d = np.atleast_1d(np.zeros(LS))

        # ###########################################################################
        # # Temperature Algorithm

        # ###########################################################################


        # Select the temperature MLD.  See the paper for a description of the
        # steps.

        # if1 testt(mldindex) == 0
        if1 = (testt==0)
        #     mixedt(mldindex) = upperdtmin(mldindex);
        #     analysis_t(mldindex) = 1;
        analysis_t[if1] = 1
        mixedt[analysis_t==1] = upperdtmin[analysis_t==1]
        #     if2 tdiff(mldindex)<0 && mixedt(mldindex) > mldepthptmp(mldindex)
        if2 = (tdiff<0) & (mixedt>mldepthptmp)
        #         mixedt(mldindex) = mldepthptmp(mldindex);
        #         analysis_t(mldindex) = 2;
        analysis_t[if1 & if2] = 2
        mixedt[analysis_t==2] = mldepthptmp[analysis_t==2]
        #     end
        #     if3 mixedt(mldindex) > mldepthptmp(mldindex)        
        if3 = (mixedt>mldepthptmp)
        #         if4 tmax(mldindex) < mldepthptmp(mldindex) && tmax(mldindex) > range
        if4 = (tmax<mldepthptmp) & (tmax>rangee)
        #             mixedt(mldindex) = tmax(mldindex);
        #             analysis_t(mldindex) = 3;
        analysis_t[if1 & (if3&if4)] = 3
        mixedt[analysis_t==3] = tmax[analysis_t==3]
        #         else (if4)
        #             mixedt(mldindex) = mldepthptmp(mldindex);
        #             analysis_t(mldindex) = 4;
        analysis_t[if1 & (if3&~if4)] = 4
        mixedt[analysis_t==4] = mldepthptmp[analysis_t==4]
        #         end
        #     end
        # else (if1)
        #     if5 abs(upperdtmin(mldindex)-mldepthptmp(mldindex)) < range && ...
        #        abs(dtandtmax(mldindex)-mldepthptmp(mldindex)) > range && ...
        #        upperdtmin(mldindex)<dtandtmax(mldindex)
        if5 = ( (abs(upperdtmin-mldepthptmp)<rangee) &
                (abs(dtandtmax-mldepthptmp)>rangee) &
                (upperdtmin<dtandtmax) )
        #         mixedt(mldindex) = upperdtmin(mldindex);
        #         analysis_t(mldindex) = 5;
        analysis_t[~if1 & if5] = 5
        mixedt[analysis_t==5] = upperdtmin[analysis_t==5]
        #     else (if5)
        #         if6 dtandtmax(mldindex) > pres(1)+range
        if6 = (dtandtmax>pres[0]+rangee)
        #            mixedt(mldindex) = dtandtmax(mldindex);
        #            analysis_t(mldindex) = 6;
        analysis_t[~if1 & ~if5 & if6] = 6
        mixedt[analysis_t==6] = dtandtmax[analysis_t==6]
        #             a = [abs(dtmax(mldindex)-upperdtmin(mldindex)) ...
        #                  abs(dtmax(mldindex)-mldepthptmp(mldindex)) ...
        #                  abs(mldepthptmp(mldindex)-upperdtmin(mldindex))];
        a = np.zeros([3,]+list(np.shape(dtmax)))
        a[0,...] = abs(dtmax-upperdtmin)
        a[1,...] = abs(dtmax-mldepthptmp)
        a[2,...] = abs(mldepthptmp-upperdtmin)
        #             if7 sum(a<range)>1
        if7 = np.sum(a<rangee,axis=0)>1
        #                 mixedt(mldindex) = upperdtmin(mldindex);
        #                 analysis_t(mldindex) = 7;
        analysis_t[~if1 & ~if5 & if6 & if7] = 7
        mixedt[analysis_t==7] = upperdtmin[analysis_t==7]
        #             end
        #             if8 mixedt(mldindex)>mldepthptmp(mldindex)
        if8 = mixedt>mldepthptmp
        #                 mixedt(mldindex) = mldepthptmp(mldindex);
        #                 analysis_t(mldindex) = 8;
        analysis_t[~if1 & ~if5 & if6 & if8] = 8
        mixedt[analysis_t==8] = mldepthptmp[analysis_t==8]
        #             end
        #         else (if6)
        #             if9 upperdtmin(mldindex)-mldepthptmp(mldindex) < range
        if9 = upperdtmin-mldepthptmp < rangee
        #                 mixedt(mldindex) = upperdtmin(mldindex);
        #                 analysis_t(mldindex) = 9;
        analysis_t[~if1 & ~if5 & ~if6 & if9] = 9
        mixedt[analysis_t==9] = upperdtmin[analysis_t==9]
        #             else (if9)
        #                 mixedt(mldindex) = dtmax(mldindex);
        #                 analysis_t(mldindex) = 10;
        analysis_t[~if1 & ~if5 & ~if6 & ~if9] = 10
        mixedt[analysis_t==10] = dtmax[analysis_t==10]
        #                 if mixedt(mldindex) > mldepthptmp(mldindex)
        if10 = mixedt > mldepthptmp
        #                     mixedt(mldindex) = mldepthptmp(mldindex);
        #                     analysis_t(mldindex) = 11;
        analysis_t[~if1 & ~if5 & ~if6 & ~if9 & if10] = 11
        mixedt[analysis_t==11] = mldepthptmp[analysis_t==11]
        #                 end
        #             end
        #         end
        #     end

        #     if mixedt(mldindex) == 0 && abs(mixedt(mldindex)-mldepthptmp(mldindex))>range
        if11 = (mixedt== 0) & (abs(mixedt-mldepthptmp)>rangee)
        #         mixedt(mldindex) = tmax(mldindex);
        #         analysis_t(mldindex) = 12;
        analysis_t[~if1 & if11] = 12
        mixedt[analysis_t==12] = tmax[analysis_t==12]
        #         if tmax(mldindex) == pres(1)
        if12 = tmax==pres[0]
        #             mixedt(mldindex) = mldepthptmp(mldindex);
        #             analysis_t(mldindex) = 13;
        analysis_t[~if1 & if11 & if12] = 13
        mixedt[analysis_t==13] = mldepthptmp[analysis_t==13]
        #         end
        #         if tmax(mldindex)>mldepthptmp(mldindex)
        if13 = tmax>mldepthptmp
        #             mixedt(mldindex) = mldepthptmp(mldindex);
        #             analysis_t(mldindex) = 14;
        analysis_t[~if1 & if11 & if13] = 14
        mixedt[analysis_t==14] = mldepthptmp[analysis_t==14]
        #         end
        #     end
        # end

        # analysis_t[if1] = 1
        # analysis_t[if1 & if2] = 2
        # analysis_t[if1 & (if3&if4)] = 3
        # analysis_t[if1 & (if3&~if4)] = 4
        # analysis_t[~if1 & if5] = 5
        # analysis_t[~if1 & ~if5 & if6] = 6
        # analysis_t[~if1 & ~if5 & if6 & if7] = 7
        # analysis_t[~if1 & ~if5 & if6 & if8] = 8
        # analysis_t[~if1 & ~if5 & ~if6 & if9] = 9
        # analysis_t[~if1 & ~if5 & ~if6 & ~if9] = 10
        # analysis_t[~if1 & ~if5 & ~if6 & ~if9 & if10] = 11
        # analysis_t[~if1 & if11] = 12
        # analysis_t[~if1 & if11 & if12] = 13
        # analysis_t[~if1 & if11 & if13] = 14

        # mixedt[analysis_t==1] = upperdtmin[analysis_t==1]
        # mixedt[analysis_t==2] = mldepthptmp[analysis_t==2]
        # mixedt[analysis_t==3] = tmax[analysis_t==3]
        # mixedt[analysis_t==4] = mldepthptmp[analysis_t==4]
        # mixedt[analysis_t==5] = upperdtmin[analysis_t==5]
        # mixedt[analysis_t==6] = dtandtmax[analysis_t==6]
        # mixedt[analysis_t==7] = upperdtmin[analysis_t==7]
        # mixedt[analysis_t==8] = mldepthptmp[analysis_t==8]
        # mixedt[analysis_t==9] = upperdtmin[analysis_t==9]
        # mixedt[analysis_t==10] = dtmax[analysis_t==10]
        # mixedt[analysis_t==11] = mldepthptmp[analysis_t==11]
        # mixedt[analysis_t==12] = tmax[analysis_t==12]
        # mixedt[analysis_t==13] = mldepthptmp[analysis_t==13]
        # mixedt[analysis_t==14] = mldepthptmp[analysis_t==14]


        # ###########################################################################
        # # Salinity Algorithm

        # # Select the salinity MLD
        # if1 testd(mldindex) == 0
        if1 = testd==0
        #     mixeds(mldindex) = upperdsmax(mldindex);
        #     analysis_s(mldindex) = 1;
        analysis_s[if1] = 1
        mixeds[analysis_s==1] = upperdsmax[analysis_s==1]
        #     if2 mixeds(mldindex) - mldepthdens(mldindex) > range
        if2 = mixeds-mldepthdens > rangee
        #         mixeds(mldindex) = mldepthdens(mldindex);
        #         analysis_s(mldindex) = 2;
        analysis_s[if1 & if2] = 2
        mixeds[analysis_s==2] = mldepthdens[analysis_s==2]
        #     end
        #     if3 upperdsmax(mldindex)-dsmin(mldindex) < 0 && mldepthdens(mldindex)-dsmin(mldindex) > 0
        if3 = (upperdsmax-dsmin<0) & (mldepthdens-dsmin>0)
        #         mixeds(mldindex) = dsmin(mldindex);
        #         analysis_s(mldindex) = 3;
        analysis_s[if1 & if3] = 3
        mixeds[analysis_s==3] = dsmin[analysis_s==3]
        #     end
        #     if4 upperdsmax(mldindex)-dsandsmin(mldindex) < range && dsandsmin(mldindex) > range
        if4 = (upperdsmax-dsandsmin<rangee) & (dsandsmin>rangee)
        #         mixeds(mldindex) = dsandsmin(mldindex);
        #         analysis_s(mldindex) = 4;
        analysis_s[if1 & if4] = 4
        mixeds[analysis_s==4] = dsandsmin[analysis_s==4]
        #     end
        #     if5 abs(mldepthdens(mldindex)-dsandsmin(mldindex)) < range && dsandsmin(mldindex) > range
        if5 = (abs(mldepthdens-dsandsmin)<rangee) & (dsandsmin>rangee)
        #         mixeds(mldindex) = dsandsmin(mldindex);
        #         analysis_s(mldindex) = 5;
        analysis_s[if1 & if5] = 5
        mixeds[analysis_s==5] = dsandsmin[analysis_s==5]
        #     end
        #     if6 mixedt(mldindex)-mldepthdens(mldindex)<0 && abs(mixedt(mldindex)-mldepthdens(mldindex))<range
        if6 = (mixedt-mldepthdens<0) & (abs(mixedt-mldepthdens)<rangee)
        #         mixeds(mldindex) = mixedt(mldindex);
        #         analysis_s(mldindex) = 6;
        analysis_s[if1 & if6] = 6
        mixeds[analysis_s==6] = mixedt[analysis_s==6]
        #         if7 abs(mixedt(mldindex)-upperdsmax(mldindex))<range && upperdsmax(mldindex)-mldepthdens(mldindex)<0
        if7 = (abs(mixedt-upperdsmax)<rangee) & (upperdsmax-mldepthdens<0)
        #             mixeds(mldindex) = upperdsmax(mldindex);
        #             analysis_s(mldindex) = 7;
        analysis_s[if1 & if6 & if7] = 7
        mixeds[analysis_s==7] = upperdsmax[analysis_s==7]
        #         end
        #     end
        #     if8 abs(mixedt(mldindex)-mldepthdens(mldindex))<abs(mixeds(mldindex)-mldepthdens(mldindex))
        if8 = abs(mixedt-mldepthdens)<abs(mixeds-mldepthdens)
        #         if9 mixedt(mldindex)>mldepthdens(mldindex)
        if9 = mixedt>mldepthdens
        #             mixeds(mldindex) = mldepthdens(mldindex);
        #             analysis_s(mldindex) = 8;
        analysis_s[if1 & if8 & if9] = 8
        mixeds[analysis_s==8] = mldepthdens[analysis_s==8]
        #         end
        #     end
        # else (if1)
        #     if10 dsandsmin(mldindex) > range
        if10 = dsandsmin>rangee
        #         mixeds(mldindex) = dsandsmin(mldindex);
        #         analysis_s(mldindex) = 9;
        analysis_s[~if1 & if10] = 9
        mixeds[analysis_s==9] = dsandsmin[analysis_s==9]
        #         if11 mixeds(mldindex)>mldepthdens(mldindex)
        if11 = mixeds>mldepthdens
        #             mixeds(mldindex) = mldepthdens(mldindex);
        #             analysis_s(mldindex) = 10;
        analysis_s[~if1 & if10 & if11] = 10
        mixeds[analysis_s==10] = mldepthdens[analysis_s==10]
        #         end
        #     else (if10)
        #         if12 dsmin(mldindex) < mldepthdens(mldindex)
        if12 = dsmin<mldepthdens
        #             mixeds(mldindex) = dsmin(mldindex);
        #             analysis_s(mldindex) = 11;
        analysis_s[~if1 & ~if10 & if12] = 11
        mixeds[analysis_s==11] = dsmin[analysis_s==11]
        #             if13 upperdsmax(mldindex)<mixeds(mldindex)
        if13 = upperdsmax<mixeds
        #                 mixeds(mldindex) = upperdsmax(mldindex);
        #                 analysis_s(mldindex) = 12;
        analysis_s[~if1 & ~if10 & if12 & if13] = 12
        mixeds[analysis_s==12] = upperdsmax[analysis_s==12]
        #             end
        #         else (if12)
        #             mixeds(mldindex) = mldepthdens(mldindex);
        #             analysis_s(mldindex) = 13;
        analysis_s[~if1 & ~if10 & ~if12 ] = 13
        mixeds[analysis_s==13] = mldepthdens[analysis_s==13]
        #             if14 upperdsmax(mldindex)<mixeds(mldindex)
        if14 = upperdsmax<mixeds
        #                 mixeds(mldindex) = upperdsmax(mldindex);
        #                 analysis_s(mldindex) = 14;
        analysis_s[~if1 & ~if10 & ~if12 & if14] = 14
        mixeds[analysis_s==14] = upperdsmax[analysis_s==14]
        #             end
        #             if15 mixeds(mldindex) == 1 #################should this be 0?
        if15 = mixeds == 1
        #                 mixeds(mldindex) = dsmin(mldindex);
        #                 analysis_s(mldindex) = 15;
        analysis_s[~if1 & ~if10 & ~if12 & if15] = 15
        mixeds[analysis_s==15] = dsmin[analysis_s==15]
        #             end
        #             if16 dsmin(mldindex) > mldepthdens(mldindex)
        if16 = dsmin>mldepthdens
        #                 mixeds(mldindex) = mldepthdens(mldindex);
        #                 analysis_s(mldindex) = 16;
        analysis_s[~if1 & ~if10 & ~if12 & if16] = 16
        mixeds[analysis_s==16] = mldepthdens[analysis_s==16]
        #             end
        #         end
        #     end
        # end

        # analysis_s[if1] = 1
        # analysis_s[if1 & if2] = 2
        # analysis_s[if1 & if3] = 3
        # analysis_s[if1 & if4] = 4
        # analysis_s[if1 & if5] = 5
        # analysis_s[if1 & if6] = 6
        # analysis_s[if1 & if6 & if7] = 7
        # analysis_s[if1 & if8 & if9] = 8
        # analysis_s[~if1 & if10] = 9
        # analysis_s[~if1 & if10 & if11] = 10
        # analysis_s[~if1 & ~if10 & if12] = 11
        # analysis_s[~if1 & ~if10 & if12 & if13] = 12
        # analysis_s[~if1 & ~if10 & ~if12 ] = 13
        # analysis_s[~if1 & ~if10 & ~if12 & if14] = 14
        # analysis_s[~if1 & ~if10 & ~if12 & if15] = 15
        # analysis_s[~if1 & ~if10 & ~if12 & if16] = 16

        # mixeds[analysis_s==1] = upperdsmax[analysis_s==1]
        # mixeds[analysis_s==2] = mldepthdens[analysis_s==2]
        # mixeds[analysis_s==3] = dsmin[analysis_s==3]
        # mixeds[analysis_s==4] = dsandsmin[analysis_s==4]
        # mixeds[analysis_s==5] = dsandsmin[analysis_s==5]
        # mixeds[analysis_s==6] = mixedt[analysis_s==6]
        # mixeds[analysis_s==7] = upperdsmax[analysis_s==7]
        # mixeds[analysis_s==8] = mldepthdens[analysis_s==8]
        # mixeds[analysis_s==9] = dsandsmin[analysis_s==9]
        # mixeds[analysis_s==10] = mldepthdens[analysis_s==10]
        # mixeds[analysis_s==11] = dsmin[analysis_s==11]
        # mixeds[analysis_s==12] = upperdsmax[analysis_s==12]
        # mixeds[analysis_s==13] = mldepthdens[analysis_s==13]
        # mixeds[analysis_s==14] = upperdsmax[analysis_s==14]
        # mixeds[analysis_s==15] = dsmin[analysis_s==15]
        # mixeds[analysis_s==16] = mldepthdens[analysis_s==16]

        # ###########################################################################
        # # Potential Density Algorithm.

        # # Select the potential density MLD
        # if1 testd(mldindex) == 0
        if1 = testd == 0
        #     mixedd(mldindex) = upperddmax(mldindex);
        #     analysis_d(mldindex) = 1;
        analysis_d[if1] = 1
        mixedd[analysis_d==1] = upperddmax[analysis_d==1]
        #     if2 mixedd(mldindex) > mldepthdens(mldindex)
        if2 = mixedd>mldepthdens
        #         mixedd(mldindex) = mldepthdens(mldindex);
        #         analysis_d(mldindex) = 2;
        analysis_d[if1 & if2] = 2
        mixedd[analysis_d==2] = mldepthdens[analysis_d==2]
        #     end
        #     aa = [abs(mixeds(mldindex)-mixedt(mldindex)) abs(upperddmax(mldindex)-mixedt(mldindex)) abs(mixeds(mldindex)-upperddmax(mldindex))];
        aa = np.zeros([3,]+list(np.shape(dtmax)))
        aa[0,...] = abs(mixeds-mixedt)
        aa[1,...] = abs(upperddmax-mixedt)
        aa[2,...] = abs(mixeds-upperddmax)
        #     if3 sum(aa<range)>1
        if3 = np.sum(aa<rangee,axis=0)>1
        #         mixedd(mldindex) = upperddmax(mldindex);
        #         analysis_d(mldindex) = 3;
        analysis_d[if1 & if3] = 3
        mixedd[analysis_d==3] = upperddmax[analysis_d==3]
        #     end
        #     if4 abs(mixeds(mldindex) - mldepthdens(mldindex)) < range && mixeds(mldindex)~=mldepthdens(mldindex)
        if4 = (abs(mixeds-mldepthdens)<rangee) & (mixeds!=mldepthdens)
        #         if5 mldepthdens(mldindex) < mixeds(mldindex)
        if5 = mldepthdens<mixeds
        #             mixedd(mldindex) = mldepthdens(mldindex);
        #             analysis_d(mldindex) = 4;
        analysis_d[if1 & if4 & if5] = 4
        mixedd[analysis_d==4] = mldepthdens[analysis_d==4]
        #         else (if5)
        #             mixedd(mldindex) = mixeds(mldindex);
        #             analysis_d(mldindex) = 5;
        analysis_d[if1 & if4 & ~if5] = 5
        mixedd[analysis_d==5] = mixeds[analysis_d==5]
        #         end
        #         if6 upperddmax(mldindex) == mldepthdens(mldindex)
        if6 = upperddmax==mldepthdens
        #             mixedd(mldindex) =  upperddmax(mldindex);
        #             analysis_d(mldindex) = 6;
        analysis_d[if1 & if4 & if6] = 6
        mixedd[analysis_d==6] = upperddmax[analysis_d==6]
        #         end
        #     end
        #     if7 mixedd(mldindex)>ddmin(mldindex) && abs(ddmin(mldindex)-mixedt(mldindex))<abs(mixedd(mldindex)-mixedt(mldindex))
        if7 = (mixedd>ddmin) & (abs(ddmin-mixedt)<abs(mixedd-mixedt))
        #         mixedd(mldindex) = ddmin(mldindex);
        #         analysis_d(mldindex) = 7;
        analysis_d[if1 & if7] = 7
        mixedd[analysis_d==7] = ddmin[analysis_d==7]
        #     end
        # else (if1)
        #     mixedd(mldindex) = mldepthdens(mldindex);
        #     analysis_d(mldindex) = 8;
        analysis_d[~if1] = 8
        mixedd[analysis_d==8] = mldepthdens[analysis_d==8]
        #     if8 mldepthptmp(mldindex)<mixedd(mldindex);
        if8 = mldepthptmp<mixedd
        #         mixedd(mldindex) = mldepthptmp(mldindex);
        #         analysis_d(mldindex) = 9;
        analysis_d[~if1 & if8] = 9
        mixedd[analysis_d==9] = mldepthptmp[analysis_d==9]
        #     end
        #     if9 upperddmax(mldindex)<mldepthdens(mldindex) && upperddmax(mldindex)>range
        if9 = (upperddmax<mldepthdens) & (upperddmax>rangee)
        #         mixedd(mldindex) =  upperddmax(mldindex);
        #         analysis_d(mldindex) = 10;
        analysis_d[~if1 & if9] = 10
        mixedd[analysis_d==10] = upperddmax[analysis_d==10]
        #     end
        #     if10 dtandtmax(mldindex) > range && dtandtmax(mldindex)<mldepthdens(mldindex)
        if10 = (dtandtmax>rangee) & (dtandtmax<mldepthdens)
        #         mixedd(mldindex) = dtandtmax(mldindex);
        #         analysis_d(mldindex) = 11;
        analysis_d[~if1 & if10] = 11
        mixedd[analysis_d==11] = dtandtmax[analysis_d==11]
        #         if11 abs(tmax(mldindex)-upperddmax(mldindex))<abs(dtandtmax(mldindex)-upperddmax(mldindex))
        if11 = (abs(tmax-upperddmax)<abs(dtandtmax-upperddmax))
        #             mixedd(mldindex) = tmax(mldindex);
        #             analysis_d(mldindex) = 12;
        analysis_d[~if1 & if10 & if11] = 12
        mixedd[analysis_d==12] = tmax[analysis_d==12]
        #         end
        #         if12 abs(mixeds(mldindex) - mldepthdens(mldindex)) < range && mixeds(mldindex)<mldepthdens(mldindex)
        if12 = (abs(mixeds-mldepthdens)<rangee) & (mixeds<mldepthdens)
        #             mixedd(mldindex) = min(mldepthdens(mldindex),mixeds(mldindex));
        #             analysis_d(mldindex) = 13;
        analysis_d[~if1 & if10 & if12] = 13
        mixedd[analysis_d==13] = np.minimum(mldepthdens,mixeds)[analysis_d==13]
        #         end
        #     end
        #     if13 abs(mixedt(mldindex)-mixeds(mldindex)) < range
        if13 = abs(mixedt-mixeds) < rangee
        #         if14 abs(min(mixedt(mldindex),mixeds(mldindex))-mixedd(mldindex)) > range
        if14 = abs(np.minimum(mixedt,mixeds)-mixedd) > rangee
        #             mixedd(mldindex) = min(mixedt(mldindex),mixeds(mldindex));
        #             analysis_d(mldindex) = 14;
        analysis_d[~if1 & if13 & if14] = 14
        mixedd[analysis_d==14] = np.minimum(mixedt,mixeds)[analysis_d==14]
        #         end
        #     end
        #     if15 mixedd(mldindex)>ddmin(mldindex) && abs(ddmin(mldindex)-mixedt(mldindex))<abs(mixedd(mldindex)-mixedt(mldindex))
        if15 = (mixedd>ddmin) & (abs(ddmin-mixedt)<abs(mixedd-mixedt))
        #         mixedd(mldindex) = ddmin(mldindex);
        #         analysis_d(mldindex) = 15;
        analysis_d[~if1 & if15] = 15
        mixedd[analysis_d==15] = ddmin[analysis_d==15]
        #     end
        #     if16 upperddmax(mldindex)==upperdsmax(mldindex) && abs(upperdsmax(mldindex)-mldepthdens(mldindex))<range
        if16 = (upperddmax==upperdsmax) & (abs(upperdsmax-mldepthdens)<rangee)
        #         mixedd(mldindex) = upperddmax(mldindex);
        #         analysis_d(mldindex) = 16;
        analysis_d[~if1 & if16] = 16
        mixedd[analysis_d==16] = upperddmax[analysis_d==16]
        #     end
        #     if17 mixedt(mldindex)==dmin(mldindex)
        if17 = mixedt==dmin
        #         mixedd(mldindex) = dmin(mldindex);
        #         analysis_d(mldindex) = 17;
        analysis_d[~if1 & if17] = 17
        mixedd[analysis_d==17] = dmin[analysis_d==17]
        #     end
        # end

        # analysis_d[if1] = 1
        # analysis_d[if1 & if2] = 2
        # analysis_d[if1 & if3] = 3
        # analysis_d[if1 & if4 & if5] = 4
        # analysis_d[if1 & if4 & ~if5] = 5
        # analysis_d[if1 & if4 & if6] = 6
        # analysis_d[if1 & if7] = 7
        # analysis_d[~if1] = 8
        # analysis_d[~if1 & if8] = 9
        # analysis_d[~if1 & if9] = 10
        # analysis_d[~if1 & if10] = 11
        # analysis_d[~if1 & if10 & if11] = 12
        # analysis_d[~if1 & if10 & if12] = 13
        # analysis_d[~if1 & if13 & if14] = 14
        # analysis_d[~if1 & if15] = 15
        # analysis_d[~if1 & if16] = 16
        # analysis_d[~if1 & if17] = 17

        # mixedd[analysis_d==1] = upperddmax[analysis_d==1]
        # mixedd[analysis_d==2] = mldepthdens[analysis_d==2]
        # mixedd[analysis_d==3] = upperddmax[analysis_d==3]
        # mixedd[analysis_d==4] = mldepthdens[analysis_d==4]
        # mixedd[analysis_d==5] = mixeds[analysis_d==5]
        # mixedd[analysis_d==6] = upperddmax[analysis_d==6]
        # mixedd[analysis_d==7] = ddmin[analysis_d==7]
        # mixedd[analysis_d==8] = mldepthdens[analysis_d==8]
        # mixedd[analysis_d==9] = mldepthptmp[analysis_d==9]
        # mixedd[analysis_d==10] = upperddmax[analysis_d==10]
        # mixedd[analysis_d==11] = dtandtmax[analysis_d==11]
        # mixedd[analysis_d==12] = tmax[analysis_d==12]
        # mixedd[analysis_d==13] = np.minimum(mldepthdens,mixeds)[analysis_d==13]
        # mixedd[analysis_d==14] = np.minimum(mixedt,mixeds)[analysis_d==14]
        # mixedd[analysis_d==15] = ddmin[analysis_d==15]
        # mixedd[analysis_d==16] = upperddmax[analysis_d==16]
        # mixedd[analysis_d==17] = dmin[analysis_d==17]
        
        # ###########################################################################
        # # Output variables
        
        # # Algorithm mlds
        # mixedtp(mldindex) = mixedt(mldindex);
        # mixedsp(mldindex) = mixeds(mldindex);
        # mixeddp(mldindex) = mixedd(mldindex);
        
        # # Theshold method mlds
        # mldepthdensp(mldindex) = mldepthdens(mldindex);
        # mldepthptmpp(mldindex) = mldepthptmp(mldindex);
        
        # # Gradient method mlds
        # gtmldp(mldindex) = pres(gtmld(mldindex));
        # gdmldp(mldindex) = pres(gdmld(mldindex));
        
        # # Record which step selected the MLD for the temperature, salinity, and
        # # potential density profiles
        # tanalysis(mldindex) = analysis_t(mldindex);
        # sanalysis(mldindex) = analysis_s(mldindex);
        # danalysis(mldindex) = analysis_d(mldindex);
        gtmld = dtmax
        gdmld = ddmin
        return mixedt, mixeds, mixedd, analysis_t, analysis_s, analysis_d, testt, testd, mldepthdens, mldepthptmp, gtmld, gdmld, upperdtmin, \
            upperdsmax, upperddmax, dtmax,dsmin,ddmin,tmax,smin,dmin,dtandtmax,dsandsmin, tdiff, ddiff
