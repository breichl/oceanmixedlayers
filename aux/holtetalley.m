% This is the original matlab/octave file from Holte and Talley, modified into a function

% This subroutine calculates the mixed layer depth (MLD) of the input profile.
% It is called by get_mld.m.

function [OUTPUT]=holtetalley(pres,temp,sal,pden)
format long

% The algorithm's parameters:
errortol = 1*10^-10; % Error tolerance for fitting a straight line to the mixed layer -- unitless
range = 25;          % Maximum separation for searching for clusters of possible MLDs -- dbar
deltad = 100;        % Maximum separation of temperature and temperature gradient maxima for identifying
% intrusions at the base of the mixed layer -- dbar
tcutoffu = .5;       % Upper temperature cutoff, used to initially classify profiles as winter or summer profiles -- degrees C
tcutoffl = -.25;     % Lower temperature cutoff, used to initially classify profiles as winter or summer profiles -- degrees C
dcutoff = -.06;      % Density cutoff, used to initially classify profiles as winter or summer profiles -- kg/m^3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate the MLD using a threshold method with de Boyer Montegut et al's
% criteria; a density difference of .03 kg/m^3 or a temperature difference
% of .2 degrees C.  The measurement closest to 10 dbar is used as the
% reference value.  The threshold MLDs are interpolated to exactly match
% the threshold criteria.

% Calculate the index of the reference value

m = length(sal);
starti = min(find((pres-10).^2==min((pres-10).^2)));
pres = pres(starti:m);
sal = sal(starti:m);
temp = temp(starti:m);
pden = pden(starti:m);
starti = 1;
m = length(sal);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Search for the first level that exceeds the potential density threshold
mldepthdens = m;
for j = starti:m
    if abs(pden(starti)-pden(j))>.03
        mldepthdens = j;
        break;
    end
end

% Interpolate to exactly match the potential density threshold
clear pdenseg presseg presinterp pdenthreshold
presseg = [pres(mldepthdens-1) pres(mldepthdens)];
pdenseg = [pden(starti)-pden(mldepthdens-1) pden(starti) - pden(mldepthdens)];
P = polyfit(presseg,pdenseg,1);
presinterp = presseg(1):.5:presseg(2);
pdenthreshold = polyval(P,presinterp);

% The potential density threshold MLD value:
mldepthdens = presinterp(max(find(abs(pdenthreshold)<.03)));

% Search for the first level that exceeds the temperature threshold
mldepthptmp = m;
for j = starti:m
    if abs(temp(starti)-temp(j))>.2
        mldepthptmp = j;
        break;
    end
end

% Interpolate to exactly match the temperature threshold
clear tempseg presseg presinterp tempthreshold
presseg = [pres(mldepthptmp-1) pres(mldepthptmp)];
tempseg = [temp(starti)-temp(mldepthptmp-1) temp(starti) - temp(mldepthptmp)];
P = polyfit(presseg,tempseg,1);
presinterp = presseg(1):.5:presseg(2);
tempthreshold = polyval(P,presinterp);

% The temperature threshold MLD value:
mldepthptmp = presinterp(max(find(abs(tempthreshold)<.2)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate the finite difference slope of the temperature, salinity and
% density profiles
clear tslope sslope dslope tslope_s sslope_s dslope_s ms

tslope = diff(temp)./diff(pres);
sslope = diff(sal)./diff(pres);
dslope = diff(pden)./diff(pres);
ms = length(tslope);

% smoothed the slope with a simple three point average using two
% neighboring points
tslope_s = (tslope(1:ms-2) + tslope(2:ms-1) + tslope(3:ms))/3;
sslope_s = (sslope(1:ms-2) + sslope(2:ms-1) + sslope(3:ms))/3;
dslope_s = (dslope(1:ms-2) + dslope(2:ms-1) + dslope(3:ms))/3;
ms = length(tslope_s);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate the MLD using a gradient method.  Following Dong et al., the gradient
% criteria are .0005 kg/m^3/dbar and .005 degrees C/dbar.  If the criteria
% are not met, the algorithm uses the temperature or density gradient extreme.
k = find( abs(dslope)>.0005 );
if any(k)
    gdmld = k(1) + 1;
else
    gdmld = min(find(abs(dslope)==max(abs(dslope))))+1;
end

l = find( abs(tslope)>.005  );
if any(l)
    gtmld = l(1) + 1;
else
    gtmld = min(find(abs(tslope)==max(abs(tslope))))+1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Fit a straight line to the profile's mixed layer. Starting at the depth
% closest to 10 dbar, use the first two points of the profile to calculate
% a straight-line least-squares fit to the mixed layer.  Increase the depth
% and the number of points used in the fit until the bottom of the
% profile. For each fit the error is calculated by summing the squared
% difference between the fit and the profile over the depth of the fit.
% This step aims to accurately capture the slope of the mixed layer, and
% not its depth.

clear errort errors errord
for j = starti+1:m
    % Fit line to temperature and calculate error
    P = polyfit(pres(starti:j),temp(starti:j),1);
    ltempfit = polyval(P,pres(starti:j));
    errort(j) = dot((temp(starti:j)-ltempfit),(temp(starti:j)-ltempfit));
    
    % Fit line to salinity and calculate error
    P = polyfit(pres(starti:j),sal(starti:j),1);
    lsalfit = polyval(P,pres(starti:j));
    errors(j) = dot((sal(starti:j)-lsalfit),(sal(starti:j)-lsalfit));
    
    % Fit line to potential density and calculate error
    P = polyfit(pres(starti:j),pden(starti:j),1);
    ldenfit = polyval(P,pres(starti:j));
    errord(j) = dot((pden(starti:j)-ldenfit),(pden(starti:j)-ldenfit));
    
    clear ltempfit lsalfit ldenfit
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Normalize the errors
errort = errort/sum(errort);
errors = errors/sum(errors);
errord = errord/sum(errord);

% Find deepest index with allowable error
upperlayert = max(find(errort < errortol));
upperlayers = max(find(errors < errortol));
upperlayerd = max(find(errord < errortol));
clear errort errors errord

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extend the mixed layer fit to the depth of the profile
P = polyfit(pres(starti:upperlayert),temp(starti:upperlayert),1);
ltempfit = polyval(P,pres(1:m));

P = polyfit(pres(starti:upperlayers),sal(starti:upperlayers),1);
lsalfit = polyval(P,pres(1:m));

P = polyfit(pres(starti:upperlayerd),pden(starti:upperlayerd),1);
ldenfit = polyval(P,pres(1:m));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Fit a straight line to the thermocline and extend the fit to the depth
% of the profile.  The extreme value of each profile's smoothed gradient
% (calculated in lines 82-84) is used to find the center of the
% thermocline.
clear dtminfit dsmaxfit ddmaxfit

dtdzmax = max(find(abs(tslope_s) == max(abs(tslope_s))))+1;

P = polyfit(pres(dtdzmax-1:dtdzmax+1),temp(dtdzmax-1:dtdzmax+1),1);
dtminfit = polyval(P,pres(1:m));

dsdzmax = max(find(abs(sslope_s) == max(max(abs(sslope_s)))))+1;
P = polyfit(pres(dsdzmax-1:dsdzmax+1),sal(dsdzmax-1:dsdzmax+1),1);
dsmaxfit = polyval(P,pres(1:m));

dddzmax = max(find(abs(dslope_s) == max(max(abs(dslope_s)))))+1;
P = polyfit(pres(dddzmax-1:dddzmax+1),pden(dddzmax-1:dddzmax+1),1);
ddmaxfit = polyval(P,pres(1:m));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate one set of possible MLD values by finding the intersection
% points of the mixed layer and thermocline fits.  If the fits do not
% intersect, the MLD value is set to 0.
upperdtmin = max(find(abs(dtminfit-ltempfit) == min(abs(dtminfit-ltempfit))));
if all(dtminfit-ltempfit>0)==1;
    upperdtmin = 0;
end
if all(-dtminfit+ltempfit>0)==1;
    upperdtmin = 0;
end

upperdsmax = max(find(abs(dsmaxfit-lsalfit) == min(abs(dsmaxfit-lsalfit))));
if all(-dsmaxfit+lsalfit>0)==1;
    upperdsmax = 0;
end
if all(dsmaxfit-lsalfit>0)==1;
    upperdsmax = 0;
end

upperddmax = max(find(abs(ddmaxfit-ldenfit) == min(abs(ddmaxfit-ldenfit))));
if all(ddmaxfit-ldenfit>0)==1;
    upperddmax = 0;
end
if all(-ddmaxfit+ldenfit>0)==1;
    upperddmax = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate the remaining possible MLD values:

% The maxima or minima of the temperature, salinity, and potential density
% profiles
tmax = max(find(temp == max(temp)));
smin = max(find(sal == min(sal)));
dmin = max(find(pden == min(pden)));

% The gradient MLD values
dtmax = gtmld;
dsmin = max(find(abs(sslope_s) == max(abs(sslope_s))))+1;
ddmin = gdmld;

% Sometimes subsurface temperature or salinity intrusions exist at the base
% of the mixed layer.  For temperature, these intrusions are
% characterized by subsurface temperature maxima located near temperature
% gradient maxima. If the two maxima are separated by less than deltad,
% the possible MLD value is recorded in dtandtmax.
dtmax2 = max(find(tslope_s == max(tslope_s)))+1;
if abs(pres(dtmax2)-pres(tmax)) < deltad;
    dtandtmax = min(dtmax2, tmax);
else
    dtandtmax = 0;
end
dsmin2 = max(find(sslope_s == min(sslope_s)))+1;
if abs(pres(dsmin2)-pres(smin)) < deltad;
    dsandsmin = min(dsmin2, smin);
else
    dsandsmin = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% To determine if the profile resembles a typical winter or summer profile,
% the temperature change across the thermocline, tdiff, is calculated and
% compared to the temperature cutoff. tdiff is calculated as the
% temperature change between the intersection of the mixed layer and thermocline fits and a
% point two depth indexes deeper.  If upperdtmin is set to 0 or at the
% bottom of the profile, the points from the thermocline fit are used
% to evaluate tdiff.
if  upperdtmin>0 && upperdtmin<(m-2)
    tdiff = temp(upperdtmin)-temp(upperdtmin+2);
else
    tdiff = temp(dtdzmax-1)-temp(dtdzmax+1);
end

% tdiff is compared to the temperature cutoffs
if tdiff > tcutoffl && tdiff<tcutoffu
    testt = 1; % winter
else
    testt = 0; % summer
end

% For salinity and potential density profiles, the potential density
% change across the pycnocline is calculated in a similar manner and
% compared to a potential density cutoff.
if upperddmax>0 && upperddmax<m-2
    ddiff = pden(upperddmax)-pden(upperddmax+2);
else
    ddiff = pden(dddzmax-1)-pden(dddzmax+1);
end
testd = testt;
if ddiff > dcutoff && tdiff > tcutoffu
    testd = 1; % winter
end
if ddiff > dcutoff && tdiff < tcutoffl
    testd = 0; % summer
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Temperature Algorithm

% Convert the possible temperature MLDs from index to pressure
if upperdtmin > 0
    upperdtmin = pres(upperdtmin);
end
tmax = pres(tmax);
if dtandtmax>0
    dtandtmax = pres(dtandtmax);
else
    dtandtmax = 0;
end
dtmax = pres(dtmax);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Select the temperature MLD.  See the paper for a description of the
% steps.
if testt == 0
    mixedt = upperdtmin;
    analysis_t = 1;
    if tdiff<0 && mixedt > mldepthptmp
        mixedt = mldepthptmp;
        analysis_t = 2;
    end
    if mixedt > mldepthptmp
        if tmax < mldepthptmp && tmax > range
            mixedt = tmax;
            analysis_t = 3;
        else
            mixedt = mldepthptmp;
            analysis_t = 4;
        end
    end
else
    if abs(upperdtmin-mldepthptmp) < range && ...
            abs(dtandtmax-mldepthptmp) > range && ...
            upperdtmin<dtandtmax
        mixedt = upperdtmin;
        analysis_t = 5;
    else
        if dtandtmax > pres(1)+range
            mixedt = dtandtmax;
            analysis_t = 6;
            a = [abs(dtmax-upperdtmin) ...
                abs(dtmax-mldepthptmp) ...
                abs(mldepthptmp-upperdtmin)];
            if sum(a<range)>1
                mixedt = upperdtmin;
                analysis_t = 7;
            end
            if mixedt>mldepthptmp
                mixedt = mldepthptmp;
                analysis_t = 8;
            end
        else
            if upperdtmin-mldepthptmp < range
                mixedt = upperdtmin;
                analysis_t = 9;
            else
                mixedt = dtmax;
                analysis_t = 10;
                if mixedt > mldepthptmp
                    mixedt = mldepthptmp;
                    analysis_t = 11;
                end
            end
        end
    end
    
    if mixedt == 0 && abs(mixedt-mldepthptmp)>range
        mixedt = tmax;
        analysis_t = 12;
        if tmax == pres(1)
            mixedt = mldepthptmp;
            analysis_t = 13;
        end
        if tmax>mldepthptmp
            mixedt = mldepthptmp;
            analysis_t = 14;
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Salinity Algorithm

% Convert the possible salinity MLDs from index to pressure
if upperdsmax>0
    upperdsmax = pres(upperdsmax);
end
dsmin = pres(dsmin);
if dsandsmin>0
    dsandsmin = pres(dsandsmin);
else
    dsandsmin = 0;
end

% Select the salinity MLD
if testd == 0
    mixeds = upperdsmax;
    analysis_s = 1;
    if mixeds - mldepthdens > range
        mixeds = mldepthdens;
        analysis_s = 2;
    end
    if upperdsmax-dsmin < 0 && mldepthdens-dsmin > 0
        mixeds = dsmin;
        analysis_s = 3;
    end
    if upperdsmax-dsandsmin < range && dsandsmin > range
        mixeds = dsandsmin;
        analysis_s = 4;
    end
    if abs(mldepthdens-dsandsmin) < range && dsandsmin > range
        mixeds = dsandsmin;
        analysis_s = 5;
    end
    if mixedt-mldepthdens<0 && abs(mixedt-mldepthdens)<range
        mixeds = mixedt;
        analysis_s = 6;
        if abs(mixedt-upperdsmax)<range && upperdsmax-mldepthdens<0
            mixeds = upperdsmax;
            analysis_s = 7;
        end
    end
    if abs(mixedt-mldepthdens)<abs(mixeds-mldepthdens)
        if mixedt>mldepthdens
            mixeds = mldepthdens;
            analysis_s = 8;
        end
    end
else
    if dsandsmin > range
        mixeds = dsandsmin;
        analysis_s = 9;
        if mixeds>mldepthdens
            mixeds = mldepthdens;
            analysis_s = 10;
        end
    else
        if dsmin < mldepthdens
            mixeds = dsmin;
            analysis_s = 11;
            if upperdsmax<mixeds
                mixeds = upperdsmax;
                analysis_s = 12;
            end
        else
            mixeds = mldepthdens;
            analysis_s = 13;
            if upperdsmax<mixeds
                mixeds = upperdsmax;
                analysis_s = 14;
            end
            if mixeds == 1 %%%%%%%%%%%%%%%%%should this be 0?
                mixeds = dsmin;
                analysis_s = 15;
            end
            if dsmin > mldepthdens
                mixeds = mldepthdens;
                analysis_s = 16;
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Potential Density Algorithm.

% Convert the possible potential density MLDs from index to pressure
if upperddmax>0
    upperddmax = pres(upperddmax);
end
dmin = pres(dmin);
ddmin = pres(ddmin);

% Select the potential density MLD
if testd == 0
    mixedd = upperddmax;
    analysis_d = 1;
    if mixedd > mldepthdens
        mixedd = mldepthdens;
        analysis_d = 2;
    end
    
    aa = [abs(mixeds-mixedt) abs(upperddmax-mixedt) abs(mixeds-upperddmax)];
    if sum(aa<range)>1
        mixedd = upperddmax;
        analysis_d = 3;
    end
    if abs(mixeds - mldepthdens) < range && mixeds~=mldepthdens
        if mldepthdens < mixeds
            mixedd = mldepthdens;
            analysis_d = 4;
        else
            mixedd = mixeds;
            analysis_d = 5;
        end
        if upperddmax == mldepthdens
            mixedd =  upperddmax;
            analysis_d = 6;
        end
    end
    if mixedd>ddmin && abs(ddmin-mixedt)<abs(mixedd-mixedt)
        mixedd = ddmin;
        analysis_d = 7;
    end
else
    mixedd = mldepthdens;
    analysis_d = 8;
    if mldepthptmp<mixedd;
        mixedd = mldepthptmp;
        analysis_d = 9;
    end
    if upperddmax<mldepthdens && upperddmax>range
        mixedd =  upperddmax;
        analysis_d = 10;
    end
    if dtandtmax > range && dtandtmax<mldepthdens
        mixedd = dtandtmax;
        analysis_d = 11;
        if abs(tmax-upperddmax)<abs(dtandtmax-upperddmax)
            mixedd = tmax;
            analysis_d = 12;
        end
        if abs(mixeds - mldepthdens) < range && mixeds<mldepthdens
            mixedd = min(mldepthdens,mixeds);
            analysis_d = 13;
        end
    end
    if abs(mixedt-mixeds) < range
        if abs(min(mixedt,mixeds)-mixedd) > range
            mixedd = min(mixedt,mixeds);
            analysis_d = 14;
        end
    end
    if mixedd>ddmin && abs(ddmin-mixedt)<abs(mixedd-mixedt)
        mixedd = ddmin;
        analysis_d = 15;
    end
    if upperddmax==upperdsmax && abs(upperdsmax-mldepthdens)<range
        mixedd = upperddmax;
        analysis_d = 16;
    end
    if mixedt==dmin
        mixedd = dmin;
        analysis_d = 17;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Output variables

% Algorithm mlds
mixedtp = mixedt;
mixedsp = mixeds;
mixeddp = mixedd;

% Theshold method mlds
mldepthdensp = mldepthdens;
mldepthptmpp = mldepthptmp;

% Gradient method mlds
gtmldp = pres(gtmld);
gdmldp = pres(gdmld);

% Find the various methods' MLD indices for computing mixed layer average
% properties
clear ta da tt dt

ta = find(pres<mixedt);
da = find(pres<mixedd);
tt = find(pres<mldepthptmp);
dt = find(pres<mldepthdens);

mixedt_ta = mean(temp(ta));
mixedd_ta = mean(temp(da));
mldepthdens_ta = mean(temp(dt));
mldepthptmp_ta = mean(temp(tt));

% Mixed layer average salinity over different MLDs
mixedt_sa = mean(sal(ta));
mixedd_sa = mean(sal(da));
mldepthdens_sa = mean(sal(dt));
mldepthptmp_sa = mean(sal(tt));

% Mixed layer average potential density over different MLDs
mixedt_da = mean(pden(ta));
mixedd_da = mean(pden(da));
mldepthdens_da = mean(pden(dt));
mldepthptmp_da = mean(pden(tt));

% Record which step selected the MLD for the temperature, salinity, and
% potential density profiles
tanalysis = analysis_t;
sanalysis = analysis_s;
danalysis = analysis_d;

% Record which step selected the MLD for the temperature, salinity, and
% potential density profiles
tanalysis = analysis_t;
sanalysis = analysis_s;
danalysis = analysis_d;
OUTPUT = [mixedtp,mixedsp,mixeddp,tanalysis,sanalysis,danalysis,testt,testd,...
       mldepthdensp, mldepthptmpp, gtmldp, gdmldp,...
       upperdtmin,upperdsmax,upperddmax,...
       dtmax,dsmin,ddmin,tmax,pres(smin),dmin,dtandtmax,dsandsmin,tdiff,ddiff];
end
