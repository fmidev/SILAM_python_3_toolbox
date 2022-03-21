import calendar
import numpy as np  #math
import datetime
from dateutil import tz


class SunTimeException(Exception):

    def __init__(self, message):
        super(SunTimeException, self).__init__(message)


class Sun: 
    """
    Approximated calculation of sunrise and sunset datetimes. Adapted from:
    https://stackoverflow.com/questions/19615350/calculate-sunrise-and-sunset-times-for-a-given-gps-coordinate-within-postgresql
    """
    def __init__(self, lat, lon):
        self._lat = lat
        self._lon = lon

    def get_sunrise_time(self, date=None):
        """
        Calculate the sunrise time for given date.
        :param lat: Latitude
        :param lon: Longitude
        :param date: Reference date. Today if not provided.
        :return: UTC sunrise datetime
        :raises: SunTimeException when there is no sunrise and sunset on given location and date
        """
        date = datetime.date.today() if date is None else date
        sr = self._calc_sun_time(date, True)
        if sr is None:
            raise SunTimeException('The sun never rises on this location (on the specified date)')
        else:
            return sr

    def get_local_sunrise_time(self, date=None, local_time_zone=tz.tzlocal()):
        """
        Get sunrise time for local or custom time zone.
        :param date: Reference date. Today if not provided.
        :param local_time_zone: Local or custom time zone.
        :return: Local time zone sunrise datetime
        """
        date = datetime.date.today() if date is None else date
        sr = self._calc_sun_time(date, True)
        if sr is None:
            raise SunTimeException('The sun never rises on this location (on the specified date)')
        else:
            return sr.astimezone(local_time_zone)

    def get_sunset_time(self, date=None):
        """
        Calculate the sunset time for given date.
        :param lat: Latitude
        :param lon: Longitude
        :param date: Reference date. Today if not provided.
        :return: UTC sunset datetime
        :raises: SunTimeException when there is no sunrise and sunset on given location and date.
        """
        date = datetime.date.today() if date is None else date
        ss = self._calc_sun_time(date, False)
        if ss is None:
            raise SunTimeException('The sun never sets on this location (on the specified date)')
        else:
            return ss

    def get_local_sunset_time(self, date=None, local_time_zone=tz.tzlocal()):
        """
        Get sunset time for local or custom time zone.
        :param date: Reference date
        :param local_time_zone: Local or custom time zone.
        :return: Local time zone sunset datetime
        """
        date = datetime.date.today() if date is None else date
        ss = self._calc_sun_time(date, False)
        if ss is None:
            raise SunTimeException('The sun never sets on this location (on the specified date)')
        else:
            return ss.astimezone(local_time_zone)

    def _calc_sun_time(self, date, isRiseTime=True, zenith=90.8):
        """
        Calculate sunrise or sunset date.
        :param date: Reference date
        :param isRiseTime: True if you want to calculate sunrise time.
        :param zenith: Sun reference zenith
        :return: UTC sunset or sunrise datetime
        :raises: SunTimeException when there is no sunrise and sunset on given location and date
        """
        # isRiseTime == False, returns sunsetTime
        try:
            day = np.ones(shape=(self._lon.size, self._lat.size)) * date.day
            month = np.ones(shape=(self._lon.size, self._lat.size)) * date.month
            year = np.ones(shape=(self._lon.size, self._lat.size)) * date.year
        except:
            day = date.day
            month = date.month
            year = date.year
        maxDayNbr = calendar.monthrange(date.year, date.month)[1]

        TO_RAD = np.pi/180.0

        # 1. first calculate the day of the year
        N1 = np.floor(275 * month / 9)
        N2 = np.floor((month + 9) / 12)
        N3 = (1 + np.floor((year - 4 * np.floor(year / 4) + 2) / 3))
        N = N1 - (N2 * N3) + day - 30

        # 2. convert the longitude to hour value and calculate an approximate time
        lngHour = self._lon / 15

        try:
            if isRiseTime:
                t = N + ((6 - lngHour) / 24)
            else: #sunset
                t = N + ((18 - lngHour) / 24)
        except:
            if isRiseTime:
                t = N + ((6 - lngHour[:,None]) / 24)
            else: #sunset
                t = N + ((18 - lngHour[:,None]) / 24)

        # 3. calculate the Sun's mean anomaly
        M = (0.9856 * t) - 3.289

        # 4. calculate the Sun's true longitude
        L = M + (1.916 * np.sin(TO_RAD*M)) + (0.020 * np.sin(TO_RAD * 2 * M)) + 282.634
        L = self._force_range(L, 360 ) #NOTE: L adjusted into the range [0,360)

        # 5a. calculate the Sun's right ascension

        RA = (1/TO_RAD) * np.arctan(0.91764 * np.tan(TO_RAD*L))
        RA = self._force_range(RA, 360 ) #NOTE: RA adjusted into the range [0,360)

        # 5b. right ascension value needs to be in the same quadrant as L
        Lquadrant  = (np.floor( L/90)) * 90
        RAquadrant = (np.floor(RA/90)) * 90
        RA = RA + (Lquadrant - RAquadrant)

        # 5c. right ascension value needs to be converted into hours
        RA = RA / 15

        # 6. calculate the Sun's declination
        sinDec = 0.39782 * np.sin(TO_RAD*L)
        cosDec = np.cos(np.arcsin(sinDec))

        # 7a. calculate the Sun's local hour angle
        try:
            cosH = (np.cos(TO_RAD*zenith) - 
                    (sinDec * np.sin(TO_RAD*self._lat))) / (cosDec * np.cos(TO_RAD*self._lat))
            if cosH > 1:
                return None     # The sun never rises on this location (on the specified date)
            if cosH < -1:
                return None     # The sun never sets on this location (on the specified date)
        except:
            cosH = (np.cos(TO_RAD*zenith) - 
                    (sinDec * np.sin(TO_RAD*self._lat[None,:]))) / (cosDec * np.cos(TO_RAD*self._lat[None,:]))
            idxNoSunRise = np.abs(cosH) > 1    # mask
            cosH[np.abs(cosH) > 1] = 0   # temporary solution to avoid numerical problems

        # 7b. finish calculating H and convert into hours

        if isRiseTime:
            H = 360 - (1/TO_RAD) * np.arccos(cosH)
        else: #setting
            H = (1/TO_RAD) * np.arccos(cosH)

        H = H / 15

        #8. calculate local mean time of rising/setting
        try:
            T = H + RA - (0.06571 * t) - 6.622
        except:
            T = H + RA[:,None] - (0.06571 * t[:,None]) - 6.622

        #9. adjust back to UTC
        try:
            UT = T - lngHour
        except:
            UT = T - lngHour[:,None]
        UT = self._force_range(UT, 24)   # UTC time in decimal format (e.g. 23.23)

        #10. Return
        hr = self._force_range(np.floor(UT), 24)
        min = np.round((UT - np.floor(UT))*60, 0).astype(np.int)
        try:
            if min == 60:
                hr += 1
                min = 0
        except:
            hr[min == 60] += 1
            min[min == 60] = 0

        #10. check corner case https://github.com/SatAgro/suntime/issues/1
        try:
            if hr == 24:
                hr = 0
                day += 1
            
                if day > maxDayNbr:
                    day = 1
                    month += 1

                    if month > 12:
                        month = 1
                        year += 1
            return datetime.datetime(year, month, day, hr, int(min), tzinfo=tz.tzutc())
        
        except:
            day[hr == 24] += 1
            hr[hr == 24] = 0
            
            idxBad = day > maxDayNbr
            month[idxBad] += 1
            day[idxBad] = 1

            year[month > 12] += 1
            month[month > 12] = 1
            
#            for y,m,d,h,mm in zip(year.flatten(),month.flatten(),day.flatten(),hr.flatten(),min.flatten()):
#                print(datetime.datetime(int(y),int(m),int(d),int(h),int(mm),tzinfo=tz.tzutc())) 

            arTime = np.array(list((datetime.datetime(int(y),int(m),int(d),int(h),int(mm),
                                                      tzinfo=tz.tzutc()) 
                                    for y,m,d,h,mm in zip(year.flatten(),month.flatten(),
                                                          day.flatten(), hr.flatten(),
                                                          min.flatten())))).reshape(year.shape)
            arTime[idxNoSunRise] = np.nan
            return arTime


    @staticmethod
    def _force_range(v, max):
        # force v to be >= 0 and < max
        try:
            if v < 0:
                return v + max
            elif v >= max:
                return v - max
        except:
            v[v<0] += max
            v[v>=max] -= max

        return v


if __name__ == '__main__':
    sun = Sun(85.0, 21.00)
    try:
        print(sun.get_local_sunrise_time())
        print(sun.get_local_sunset_time())

    # On a special date in UTC

        abd = datetime.date(2014, 1, 3)
        abd_sr = sun.get_local_sunrise_time(abd)
        abd_ss = sun.get_local_sunset_time(abd)
        print(abd_sr)
        print(abd_ss)
    except SunTimeException as e:
        print("Error: {0}".format(e))

