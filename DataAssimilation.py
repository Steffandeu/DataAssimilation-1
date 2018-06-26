#!/opt/rh/rh-python35/root/bin/python3
import pandas as pd
import math
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import glob
import ipdb
from datetime import datetime, timedelta
import os
import shutil
import threading
import sys
import re
import subprocess
import time
from sys import exit
import platform
from time import sleep 

############################################################################
# Created in Mar 2018 by Minzheng Wang
# This is the control script to perform 3DVAR data assimilation based on BoM
# AWS and Radar data for WRF forecast.
# This program is called from auto.py, which is deployed in crontab to run 
# automatically when new data is available. If this program fails, relative 
# files will be restored to run a simple forecast without DA in auto.py
# In this demonstrative version, part of the code is changed for 
# confidentiality reason.
############################################################################


# read settings from control file
def DAsimulation_info():
    fid = open('simulation_namelist')
    for line in fid:
        if (line.split()[0] == 'run_27_manually'):
            run_27_manually = line.split()[2]
            if run_27_manually == 'True':
                run_27_manually = True
            elif run_27_manually == 'False':
                run_27_manually = False
        if (line.split()[0] == 'SIM_LEN'):
            total_hours = line.split()[2]
        if (line.split()[0] == 'DAdomains'):
            DAdomains = line.split()[2]
            DAdomains = [int(x) for x in DAdomains.split(',')]
        if (line.split()[0] == 'useRadar'):
            useRadar = line.split()[2]
            if useRadar == 'True':
                useRadar = True
            elif useRadar == 'False':
                useRadar = False
        if (line.split()[0] == 'useAWS'):
            useAWS = line.split()[2]
            if useAWS == 'True':
                useAWS = True
            elif useAWS == 'False':
                useAWS = False
        if (line.split()[0] == 'do_EOF'):
            do_EOF = line.split()[2]
            if do_EOF == 'True':
                do_EOF = True
            elif do_EOF == 'False':
                do_EOF = False
        if (line.split()[0] == 'radar_prep_method'):
            radar_prep_method = line.split()[2]
        if (line.split()[0] == 'base_dir'):
            base_dir = line.split()[2]
        if (line.split()[0] == 'TAG'):
            tag = line.split()[2]
        if (line.split()[0] == 'AWS_dir'):
            AWS_dir = line.split()[2]
        if (line.split()[0] == 'Radar_dir'):
            Radar_dir = line.split()[2]
        if (line.split()[0] == 'RUN_DIR'):
            RUN_DIR = line.split()[2]
        if (line.split()[0] == 'RadarDA_inter'):
            RadarDA_inter = line.split()[2]
        if (line.split()[0] == 'restart_DA'):
            restart_DA = line.split()[2]
            if restart_DA == 'True':
                restart_DA = True
            elif restart_DA == 'False':
                restart_DA = False
        if (line.split()[0] == 'restart_time'):
            restart_time_str = line.split()[2]
            restart_time = pd.to_datetime(
                restart_time_str, format='%Y-%m-%d_%H:%M:%S')
    fid.close()
    return (run_27_manually, int(total_hours)-1, DAdomains, useRadar, useAWS, do_EOF,
            int(radar_prep_method), int(RadarDA_inter), RUN_DIR, base_dir, tag, AWS_dir, Radar_dir, restart_DA, restart_time)


# Father class of all models, contain basic methods: prepare files, write namelist and execute.
class Model(object):
    def __init__(self):
        (_, _, _, _, _, _, _, _, RUN_DIR, base_dir, tag,
         AWS_dir, Radar_dir, _, _) = DAsimulation_info()
        self.env = {
            'RUN_DIR': RUN_DIR,
            'WRF_domains': base_dir+'WRF_domains/'+tag+'/',
            'WRFDA_domains': base_dir+'WRFDA_domains/'+tag+'/',
            'obs_dir': base_dir+'WRFDA_domains/'+tag+'/obsproc/',
            'wrfvar_dir': base_dir+'WRFDA_domains/'+tag+'/wrfvar',
            'BC_dir': base_dir+'WRFDA_domains/'+tag+'/update_bc/',
            'AWS_dir': AWS_dir,
            'Radar_dir': Radar_dir,
            'lat': -38.076,
            'lon': 144.901,
            'NLEV': 99,
            'NLAT': [99, 99, 99],   # d01, d02, d03
            'NLON': [99, 99, 99],
            'RES': [99, 99, 99]
            }

    def prep_file(self): 
        self.ln_or_cp = list(self.ln_or_cp)
        self.input_files = list(self.input_files)
        self.output_files = list(self.output_files)
        nfile = len(self.input_files)
        RUN_DIR = self.env['RUN_DIR']
        if nfile > 0:
            for i in range(nfile):
                ifile = self.input_files[i]
                ofile = self.output_files[i]
                if not os.path.isabs(ifile):
                    print('warning file does not have absolute path')
                    import ipdb
                    ipdb.set_trace()
                if not os.path.isabs(ofile):
                    print('warning file does not have absolute path')
                    import ipdb
                    ipdb.set_trace()
                if self.ln_or_cp[i] == 'cp':
                    option = 'cp'
                    try:
                        os.remove(ofile)
                    except FileNotFoundError:
                        pass
                    shutil.copy(ifile, ofile)
                    hk.write_log(RUN_DIR, str(datetime.now())+' prep_file:')
                    hk.write_log(
                        RUN_DIR, option+' {} {}'.format(self.input_files[i], self.output_files[i]))
                elif self.ln_or_cp[i] == 'ln':
                    option = 'ln -s'
                    try:
                        os.remove(ofile)
                    except FileNotFoundError:
                        pass
                    os.symlink(ifile, ofile)
                    hk.write_log(RUN_DIR, str(datetime.now())+' prep_file:')
                    hk.write_log(
                        RUN_DIR, option+' {} {}'.format(self.input_files[i], self.output_files[i]))
                else:
                    print('wrong command: '+self.ln_or_cp[i])
                    hk.write_log(RUN_DIR, str(datetime.now())+' prep_file:')
                    hk.write_log(RUN_DIR, 'wrong command: '+self.ln_or_cp[i]+' {} {}'.format(
                        self.input_files[i], self.output_files[i]))
                    os._exit(0)

    def create_namelist(self):
        template = []
        with open(self.template_filename, 'r') as template_fid:
            for line in template_fid:
                template.append(line)
        pattern = re.compile("|".join(self.replace_dict.keys()))
        lines2write = []
        for line in template:
            line = pattern.sub(lambda m: str(
                self.replace_dict[re.escape(m.group(0))]), line)
            lines2write.append(line)
        with open(self.namelist_out, 'w') as fid_write:
            fid_write.writelines(''.join(lines2write))
        fid_write.close()

    def run_model(self):  
        print(f'about to run {self.commandname}')
        run_command = subprocess.Popen(
            './'+self.commandname, cwd=self.work_dir)
        if self.ifwait:
            run_command.wait()

    def execute_all(self):
        self.prep_file()
        self.create_namelist()
        self.run_model()
        pass


# Radar data pre-process
class RadarPrep(Model):
    def __init__(self, current_time, radar_prep_method):
        Model.__init__(self)
        self.current_time = current_time
        self.radar_prep_method = radar_prep_method
        self.clearSky = False

    def write_wrfda_format(self, str_date, dbz, rad_vel, heights, lats, lons, rad_lat, rad_lon, rad_elev):
        outputfile = self.env['wrfvar_dir']+'/ob.radar'
        fid = open(outputfile, 'w')
        fake_num_locs = 'xxxxxx'
        fid.write('TOTAL NUMBER =  1\n')
        fid.write('#-----------------#\n')
        fid.write('\n')
        fid.write('RADAR'.rjust(5)+'  '+'    '.rjust(12)+"%8.3f" % rad_lon+'  '+"%8.3f" % rad_lat+'  ' +
                  "%8.1f" % rad_elev+'  '+str_date.rjust(19)+str(fake_num_locs).rjust(6)+str(10).rjust(6)+'\n')
        fid.write(
            '#-------------------------------------------------------------------------------#\n')
        fid.write('\n')
        skipped_count = 0
        for i in range(np.shape(dbz)[1]):
            for j in range(np.shape(dbz)[2]):
                the_lat = lats[i, j]
                the_lon = lons[i, j]
                if (np.ma.is_masked(np.sum(dbz[:, i, j])) == True):
                    skipped_count += 1  
                    continue  
                if np.isnan(the_lat) == True:
                    skipped_count += 1
                    continue
                if np.isnan(the_lon) == True:
                    skipped_count += 1
                    continue
                if np.ma.is_masked(the_lat) == True:
                    skipped_count += 1
                    continue
                if np.ma.is_masked(the_lon) == True:
                    skipped_count += 1
                    continue
                vel_data = ''
                dbz_data = ''
                full_data = ''
                level_count = 0
                for lev in range(np.shape(dbz)[0]):
 
                    ###################################
                    # omitting Radar data pre-process #
                    ################################### 
 
                    full_data += vel_data+dbz_data
                    vel_data = ''
                    dbz_data = ''
                header = ''
                header += 'CV-128 RADAR'.ljust(12)
                header += '   '
                header += str_date.rjust(19)
                header += '  '
                header += "%12.3f" % the_lat
                header += '  '
                header += "%12.3f" % the_lon
                header += '  '
                header += "%8.1f" % rad_elev
                header += '  '
                header += str(level_count).rjust(6)+'\n'
                fid.write(header)
                fid.write(full_data)
        num_locs = np.shape(dbz)[1]*np.shape(dbz)[2]
        print('skipped count = '+str(skipped_count))
        print('total_count = '+str(num_locs-skipped_count))
        fid.close()
        os_command = "sed -i 's/xxxxxx/" + \
            str(num_locs-skipped_count).rjust(6)+"/g' "+outputfile
        os.system(os_command)

    def get_closest_point(self, dbz, lats, lons): 
        # omitting radar pre-process method 1
        pass
    def downsampling(self, dbz1, rv1, lats1, lons1):
        # omitting radar pre-process method 2
        pass

    def bulk_avg(self):
        # omitting radar pre-process method 3
        pass
        
    def run_prep(self):
        method = self.radar_prep_method
        data_dir = self.env['Radar_dir']
        tagettime = self.current_time
        dateNtime = tagettime.strftime('%Y%m%d_%H%M').split('_')
        radar_file = glob.glob(data_dir+dateNtime[0]+'/'+dateNtime[1]+'*.h5')
        if len(radar_file) == 0:
            tagettime = self.current_time-timedelta(minutes=6)
            dateNtime = tagettime.strftime('%Y%m%d_%H%M').split('_')
            radar_file = glob.glob(
                data_dir+dateNtime[0]+'/'+dateNtime[1]+'*.h5')
            if len(radar_file) == 0:
                tagettime = self.current_time-timedelta(minutes=6)
                dateNtime = tagettime.strftime('%Y%m%d_%H%M').split('_')
                radar_file = glob.glob(
                    data_dir+dateNtime[0]+'/'+dateNtime[1]+'*.h5')
                if len(radar_file) == 0:
                    print('file missing')
                    os._exit(0)
        print('processing data: ' +
              radar_file[0].split('/')[-2]+'/'+radar_file[0].split('/')[-1])
        grid = convert_to_cartesian(radar_file[0])
        lats_ori = grid.get_point_longitude_latitude()[1]
        lons_ori = grid.get_point_longitude_latitude()[0]
        heights = grid.z['data']
        rad_lat = grid.radar_latitude['data']
        rad_lon = grid.radar_longitude['data']
        # rad_elev=grid.radar_altitude['data']
        rad_elev = 44   # 44 is more accurate.
        rv_ori = grid.fields['velocity']['data']
        dbz_ori = grid.fields['reflectivity']['data']
        if method == 0:
            dbz = dbz_ori
            rad_vel = rv_ori
            lats = lats_ori
            lons = lons_ori
        elif method == 1:
            dbz, rad_vel, lats, lons = self.downsampling(
                dbz_ori, rv_ori, lats_ori, lons_ori)
        elif method == 2:
            dbz, rad_vel, lats, lons = self.bulk_avg()
        elif method == 3:
            dbz = self.get_closest_point(dbz, lats, lons)
        # ipdb.set_trace()  # write a png for diagnosis
        # pl.pcolor(lons,lats,dbz[0],vmin=-20,vmax=30)
        # pl.colorbar()
        # pl.savefig('dbz_downsampled_'+tagettime.strftime('%Y-%m-%d_%H:%M:00')+'.png')
        self.write_wrfda_format(tagettime.strftime(
            '%Y-%m-%d_%H:%M:00'), dbz, rad_vel, heights, lats, lons, rad_lat, rad_lon, rad_elev)
        ifile = self.env['WRFDA_domains'] + '/wrfvar/ob.radar'
        ofile = self.env['WRFDA_domains'] + '/wrfvar_d01/ob.radar'
        shutil.copy(ifile, ofile)
        ofile = self.env['WRFDA_domains'] + '/wrfvar_d02/ob.radar'
        shutil.copy(ifile, ofile)
        ofile = self.env['WRFDA_domains'] + '/wrfvar_d03/ob.radar'
        shutil.copy(ifile, ofile)


# ground observation pre-process        
class ObsProc(Model):
    def __init__(self, current_time, domain, do_EOF):
        self.current_time_str = current_time.strftime('%Y%m%d%H%M')
        self.current_time = current_time
        self.time_beg = self.current_time-timedelta(minutes=30)
        self.time_end = self.current_time+timedelta(minutes=30)
        Model.__init__(self)
        self.domain = domain
        self.do_EOF = do_EOF
        self.ln_or_cp = []
        self.input_files = []
        self.output_files = []
        self.template_filename = self.env['RUN_DIR'] + \
            '/template/DA_obsproc_namelist.obsproc.templ'
        self.namelist_out = self.env['obs_dir']+'namelist.obsproc'
        self.commandname = 'obsproc.exe'
        self.ifwait = 1
        self.work_dir = self.env['obs_dir']
        self.replace_dict = {'YYYYMMDDHHMM': self.current_time_str,
                             'XXXXXXXXXXXXXXXXXXX': self.time_beg.strftime('%Y-%m-%d_%H:%M:00'),
                             'YYYYYYYYYYYYYYYYYYY': self.current_time.strftime('%Y-%m-%d_%H:%M:00'),
                             'ZZZZZZZZZZZZZZZZZZZ': self.time_end.strftime('%Y-%m-%d_%H:%M:00'),
                             'LATLATLAT': self.env['lat'],
                             'LONLONLON': self.env['lon'],
                             'NLAT': self.env['NLAT'][domain-1],
                             'NLON': self.env['NLON'][domain-1],
                             'RES': self.env['RES'][domain-1]}

    def get_BoM_data(self):
        stamps = pd.date_range(start=self.time_beg,
                               end=self.time_end,  freq='15Min')
        timestr5 = ['']*stamps.size
        for i in range(stamps.size):
            stpstr = str(stamps[i])
            timenogap = stpstr[0:4]+stpstr[5:7] + \
                stpstr[8:10]+stpstr[11:13]+stpstr[14:16]
            timestr5[i] = timenogap
        filelist = []
        for i in range(stamps.size):
            filei = glob.glob(self.env['AWS_dir']+'*'+timestr5[i]+'*.axf')
            try:
                filelist.append(filei[0])
            except:
                continue
        filelist.sort()
        BoMData = pd.read_csv(
            filelist[0], sep=', ', header=1, skipfooter=1, usecols=range(10), engine='python')

        #################################
        # omitting BoM data pre-process #
        #################################            
            
        dataOut = BoMData.loc[:, ['ID_num', 'ID_name[6]', 'Lat', 'Lon',
                                  'Height (m)', 'time_stamp', 'Wdir', 'Wspd', 'T_DB', 'DP']]
        dataOut.to_csv(self.env['obs_dir']+'obs'+str(self.current_time_str) +
                       '.csv', sep=',', header=True, index=False)

    def time_stamp(self, df):
        timeString = str(df['Date'])+' '+str(df['Time']).zfill(4)
        return pd.to_datetime(timeString, format='%Y%m%d %H%M')

    def run_obsproc(self):
        if self.do_EOF:
            ifile = self.env['obs_dir'] + '/obs_6hours.csv'
            ofile = self.env['obs_dir'] + 'obs' + \
                str(self.current_time_str) + '.csv'
            shutil.copy(ifile, ofile)
        else:
            self.get_BoM_data()
        run_command = [self.env['obs_dir'] +
                       'write_little_r.exe', str(self.current_time_str)]
        print(f"running little_r: {' '.join(run_command)}")
        run_little_r = subprocess.Popen(run_command, cwd=self.env['obs_dir'])
        run_little_r.wait()
        print('running write_little_r.exe')
        self.execute_all()
        obs_in = self.env['obs_dir']+'obs_gts_' + \
            str(self.current_time.strftime('%Y-%m-%d_%H:%M:00'))+'.3DVAR'
        obs_out = self.env['wrfvar_dir']+'_d0'+str(self.domain)+'/ob.ascii'
        while not os.path.isfile(obs_in):
            print(f'waiting for obs ascii file {obs_in}.')
            sleep(20)
        try:
            os.remove(obs_out)
        except FileNotFoundError:
            pass
        shutil.copy(obs_in, obs_out)


# data assimilation model        
class DAWRFvar(Model):
    def __init__(self, current_time_str, fg_file, domain):
        self.domain = domain
        self.current_time_str = current_time_str
        self.current_time = pd.to_datetime(
            current_time_str, format='%Y%m%d%H%M')
        self.time_beg = self.current_time-timedelta(minutes=30)
        self.time_end = self.current_time+timedelta(minutes=30)
        Model.__init__(self)
        ana_yyyy = self.current_time_str[0:4]
        ana_mm = self.current_time_str[4:6]
        ana_dd = self.current_time_str[6:8]
        ana_hh = self.current_time_str[8:10]
        self.replace_dict = {'YYYYMMDDHHMM': self.current_time_str,
                             'XXXXXXXXXXXXXXXXXXX': str(self.time_beg.strftime('%Y-%m-%d_%H:%M:%S')),
                             'YYYYYYYYYYYYYYYYYYY': str(self.current_time.strftime('%Y-%m-%d_%H:%M:%S')),
                             'ZZZZZZZZZZZZZZZZZZZ': str(self.time_end.strftime('%Y-%m-%d_%H:%M:%S')),
                             'ana_yyyy': ana_yyyy,
                             'ana_mm': ana_mm,
                             'ana_dd': ana_dd,
                             'ana_hh': ana_hh,
                             'NLEV': self.env['NLEV'],
                             'NLAT': self.env['NLAT'][domain-1],
                             'NLON': self.env['NLON'][domain-1],
                             'RESdx': self.env['RES'][domain-1]*1000,
                             'RESdy': self.env['RES'][domain-1]*1000}
        self.work_dir = self.env['wrfvar_dir']+'_d0'+str(domain)
        fg_in = self.env['WRF_domains'] + fg_file
        fg_out = self.env['wrfvar_dir']+'_d0'+str(domain)+'/fg'
        fg_opt = 'ln'
        be_in = self.env['WRFDA_domains']+'gen_be/' + \
            'gen_be5_cv5_d0'+str(domain)+'/be.dat'
        be_out = self.env['wrfvar_dir']+'_d0'+str(domain)+'/be.dat'
        be_opt = 'cp'
        self.input_files = [fg_in, be_in]
        self.output_files = [fg_out, be_out]
        self.ln_or_cp = [fg_opt, be_opt]
        self.template_filename = self.env['RUN_DIR'] + \
            '/template/DA_namelist.input.templ'
        self.namelist_out = self.env['wrfvar_dir'] + \
            '_d0'+str(domain)+'/namelist.input'
        self.commandname = 'da_wrfvar.exe'
        self.ifwait = 1

    def run_da_paral(self):
        self.prep_file()
        self.create_namelist()
        run_command = subprocess.Popen(
            './'+self.commandname, cwd=self.work_dir, shell=True)
        return run_command

    def run_model(self):
        cpus_DA = 12
        self.prep_file()
        self.create_namelist()
        dt_wrf_begin = datetime.now()
        (tag, wps_dir, wrf_dir, run_dir, out_dir, sim_len, out_freq, core, max_dom,
            hosting, queue, cpus, gfs_res, gfs_freq, time_limit, write_input) = hk.simulation_info()
        hostname = platform.node()
        qsub_run()


def check_DA_prog(rsl_file):
    os_command = 'tail -n 7 ' + rsl_file + \
        ' | grep -m 1 "WRF-Var completed successfully"'
    try:
        log = subprocess.check_output(os_command, shell=True).decode('utf-8')
        if 'successfully' in log:
            completed = True
    except:
        completed = False
    return completed


# boundary condition updating model   
class UpdateBC(Model):
    def __init__(self, current_time, bc_option, domain):
        Model.__init__(self)
        wrf_bdy_name = './wrfbdy_d01'
        wrf_input_name = './wrfinput_d0'+str(domain)
        if bc_option == 'lateral':
            lateral_or_not = '.true.'
            lower_or_not = '.false.'
            da_name = './wrfvar_output'
        elif bc_option == 'lower':
            lateral_or_not = '.false.'
            lower_or_not = '.true.'
            da_name = './fg'
        self.replace_dict = {'da_name': da_name,
                             'wrf_bdy_name': wrf_bdy_name,
                             'wrf_input_name': wrf_input_name,
                             'lateral_or_not': lateral_or_not,
                             'lower_or_not': lower_or_not,
                             'ndomain': domain}
        self.template_filename = self.env['RUN_DIR'] + \
            '/template/DA_update_bc_parame.in.templ'
        self.namelist_out = self.env['BC_dir']+'parame.in'
        if bc_option == 'lateral':
            # only update d01's lateral BC
            da_file_in = self.env['wrfvar_dir']+'_d01/wrfvar_output_d01'
            da_file_out = self.env['BC_dir']+'wrfvar_output'
            da_file_opt = 'ln'
            file2_in = self.env['WRF_domains']+'wrfbdy_d01'
            file2_out = self.env['BC_dir']+'wrfbdy_d01'
            file2_opt = 'cp'
        elif bc_option == 'lower':
            # work out wrf_3dvar_input timestamp (second could be random: wrf_3dvar_input_d03_2018-05-03_18:36:06)
            files = glob.glob(
                self.env['WRF_domains']+'wrf_3dvar_input_d0'+str(domain)+'*')
            tmstps = [pd.to_datetime(
                x[-19:], format='%Y-%m-%d_%H:%M:%S') for x in files]
            tmdiffsec = [abs(x-current_time).seconds for x in tmstps]
            right_file = np.array(files)[(np.array(tmdiffsec) == min(
                tmdiffsec)) & (np.array(tmdiffsec) <= 60)]
            if right_file.shape[0] == 0:
                print('file error in bc: '+'wrf_3dvar_input_d0'+str(domain))
                print('looking for: '+current_time.strftime('%Y-%m-%d_%H:%M:%S'))
            da_file_in = right_file[0]
            da_file_out = self.env['BC_dir']+'fg'
            da_file_opt = 'ln'
            file2_in = self.env['WRF_domains'] + \
                '/backup/wrfinput_d0'+str(domain)
            file2_out = self.env['BC_dir']+'wrfinput_d0'+str(domain)
            file2_opt = 'cp'
        self.input_files = [da_file_in, file2_in]
        self.output_files = [da_file_out, file2_out]
        self.ln_or_cp = [da_file_opt, file2_opt]
        # execute
        self.work_dir = self.env['BC_dir']
        self.commandname = 'da_update_bc.exe'
        self.ifwait = 1


# weather forecast model        
class WRF(Model):
    def __init__(self, time_start, time_end, if_3dvar, DAdomains):
        self.time_end = time_end
        time_start_round = pd.Timestamp(time_start[0]).round(freq='min')
        self.current_time = time_start_round
        Model.__init__(self)
        [yr01, mo01, d01, h01, mi01, se01] = time_start[0].strftime(
            '%Y_%m_%d_%H_%M_%S').split('_')
        [yr02, mo02, d02, h02, mi02, se02] = time_start[1].strftime(
            '%Y_%m_%d_%H_%M_%S').split('_')
        [yr03, mo03, d03, h03, mi03, se03] = time_start[2].strftime(
            '%Y_%m_%d_%H_%M_%S').split('_')
        self.addition_time = 0
        if if_3dvar:
            self.addition_time = 3        # extra time to ensure the last output will be produced
            [YYYY, MM, DD, HH, MI, SE] = (
                time_end+timedelta(minutes=self.addition_time)).strftime('%Y_%m_%d_%H_%M_%S').split('_')
        else:
            [YYYY, MM, DD, HH, MI, SE] = (time_end).strftime(
                '%Y_%m_%d_%H_%M_%S').split('_')
        
        ###################################################
        # omitting some hard-coded settings for WRF model #
        ###################################################
        
        self.replace_dict = {
            'yr01': yr01, 'yr02': yr02, 'yr03': yr03,
            'mo01': mo01, 'mo02': mo02, 'mo03': mo03,
            'd01': d01, 'd02': d02, 'd03': d03,
                        'h01': h01, 'h02': h02, 'h03': h03,
                        'mi01': mi01, 'mi02': mi02, 'mi03': mi03,
                        'se01': se01, 'se02': se02, 'se03': se03,
                        'YYYY': YYYY,
                        'MM': MM,
                        'DD': DD,
                        'HH': HH,
                        'MI': MI,
                        'SE': SE,
                        'GFS_FREQ': data_interval,
                        'if_3dvar': str(if_3dvar),
                        'min_3dvar': min_3dvar,
                        'dbz_min': dbz_min,
                        'DOM': 3
            }
        self.template_filename = self.env['RUN_DIR'] + \
            'template/namelist_10min.input.template'
        self.namelist_out = self.env['WRF_domains'] + 'namelist.input'
        self.input_files = []
        self.output_files = []
        self.ln_or_cp = []
        for domain in DAdomains:
            wrfinput_in = self.env['wrfvar_dir']+'_d0' + \
                str(domain)+'/wrfvar_output_d0'+str(domain)
            wrfinput_out = self.env['WRF_domains']+'wrfinput_d0'+str(domain)
            wrfinput_opt = 'cp'
            self.input_files.append(wrfinput_in)
            self.output_files.append(wrfinput_out)
            self.ln_or_cp.append(wrfinput_opt)
        wrfbdy_in = self.env['BC_dir']+'wrfbdy_d01'
        wrfbdy_out = self.env['WRF_domains']+'wrfbdy_d01'
        wrfbdy_opt = 'cp'
        self.input_files.append(wrfbdy_in)
        self.output_files.append(wrfbdy_out)
        self.ln_or_cp.append(wrfbdy_opt)
        self.work_dir = self.env['WRF_domains']

    def run_model(self):
        dt_wrf_begin = datetime.now()
        (tag, wps_dir, wrf_dir, run_dir, out_dir, sim_len, out_freq, core, max_dom,
            hosting, queue, cpus, gfs_res, gfs_freq, time_limit, write_input) = hk.simulation_info()
        hostname = platform.node()
        qsub_run()
        completed = False
        percent_old = 0
        time_wait = 0
        while not completed:
            sleep(60)
            the_dir = wrf_dir + tag + '/'
            percent_done, completed = self.check_run_progress_arw()
            # hack for mis-reads of file
            if percent_done == -99:
                percent_done = percent_old
            percent_old = percent_done
            msg = '{} % {}'.format(percent_done, time_wait)
            hk.write_log(run_dir, msg)
            hk.write_status_log(run_dir, msg)
            time_wait += 1
        dt_wrf_end = datetime.now()
        wrf_time = dt_wrf_end - dt_wrf_begin
        hk.write_log(run_dir, 'wrf time= ' + str(wrf_time.total_seconds()))
        # files can still be writing out to gluster
        msg = 'sleeping for 20 seconds to make sure WRF is really finsihed'
        hk.write_log(run_dir, msg)
        sleep(20)
        hk.write_status_log(run_dir, 'wrf completed')

    def check_run_progress_arw(self):
        """
        Check the rsl file and check to see what % of the way through
        the simulation we are
        """
        rsl_file = self.env['WRF_domains']+'rsl.error.0000'
        current_date = self.current_time
        sim_len = self.run_minutes+self.addition_time
        os_command = 'tail -n 7 ' + rsl_file + ' | grep -m 1 "Timing for main"'
        try:
            log = subprocess.check_output(
                os_command, shell=True).decode('utf-8')
            time_done = log.split()[4]
            time_done_datetime = datetime.strptime(
                time_done, '%Y-%m-%d_%H:%M:%S')
            end_date = self.time_end
            time_diff = time_done_datetime - current_date
            time_diff_minutes = time_diff.total_seconds() / (60.)
            percent_done = (time_diff_minutes / (sim_len - 1)) * 100
        except:
            percent_done = -99
        os_command = 'tail -n 7 ' + rsl_file + ' | grep -m 1 "SUCCESS COMPLETE WRF"'
        try:
            log = subprocess.check_output(
                os_command, shell=True).decode('utf-8')
            if 'SUCCESS' in log:
                SUCCESS = True
        except:
            SUCCESS = False
        return int(percent_done), SUCCESS


# quality control model        
class EOFQC(Model):
    def __init__(self):
        Model.__init__(self)

    def time_stamp(self, df):
        timeString = str(df['Date'])+' '+str(df['Time']).zfill(4)
        return pd.to_datetime(timeString, format='%Y%m%d %H%M')

    def get_U(self, r):
        if (r['Wspd'] != -9999)and(r['Wdir'] != -9999):
            U = r['Wspd']*math.cos((270.0-r['Wdir'])*math.pi/180.0)
        else:
            U = np.nan
        return U

    def get_V(self, r):
        if (r['Wspd'] != -9999)and(r['Wdir'] != -9999):
            V = r['Wspd']*math.sin((270.0-r['Wdir'])*math.pi/180.0)
        else:
            V = np.nan
        return V

    def EOF_func(self, analysis_time):
        # get obs data from the past 30 hours (about 60 time steps)
        timestart = analysis_time-timedelta(hours=30)
        stamps = pd.date_range(
            start=timestart, end=analysis_time,  freq='15Min')
        timestr5 = ['']*stamps.size
        for i in range(stamps.size):
            stpstr = str(stamps[i])  # format: 2018-03-05 01:00:00
            timenogap = stpstr[0:4]+stpstr[5:7] + \
                stpstr[8:10]+stpstr[11:13]+stpstr[14:16]
            timestr5[i] = timenogap
        filelist = []
        for i in range(stamps.size):
            filei = glob.glob(self.env['AWS_dir']+'*'+timestr5[i]+'*.axf')
            try:
                filelist.append(filei[0])
            except:
                continue
        filelist.sort()
        BoMData = pd.read_csv(
            filelist[0], sep=', ', header=1, skipfooter=1, usecols=range(10), engine='python')
        for ifile in filelist:
            BoMData1 = pd.read_csv(
                ifile, sep=', ', header=1, skipfooter=1, usecols=range(10), engine='python')
            BoMData = pd.concat([BoMData1, BoMData])
            del(BoMData1)
        BoMData = BoMData.drop_duplicates(
            subset=None, keep='first', inplace=False)
        BoMData['time_stamp'] = ''
        BoMData['time_stamp'] = BoMData.apply(
            lambda r: self.time_stamp(r), axis=1)
        infodtype = {'Height (m)': 'float', 'ID_num': 'int'}
        AWSInfo = pd.read_csv(
            self.env['obs_dir']+'AWSInfo_trim.csv', sep=',', engine='python', dtype=infodtype)
        BoMData = pd.merge(
            BoMData, AWSInfo[['ID_num', 'Height (m)']], how='left')
        # select stations within WRF domain
        BoMstat_d01 = pd.read_csv(
            self.env['obs_dir']+'BoM_AWS_IdLatLonElev_d01.csv', sep=',', engine='python')
        BoMstat_d03 = pd.read_csv(
            self.env['obs_dir']+'BoM_AWS_IdLatLonElev_d03.csv', sep=',', engine='python')
        AWS_domain = pd.merge(
            BoMstat_d01, BoMData[['ID_num', 'time_stamp', 'Wspd', 'Wdir', 'T_DB']])
        AWS_ID = AWS_domain['ID_num'].unique()
        # can't do EOF on wdir, convert wspd/wdir to u/v
        AWS_domain['u'] = ''
        AWS_domain['v'] = ''
        AWS_domain['u'] = AWS_domain.apply(lambda r: self.get_U(r), axis=1)
        AWS_domain['v'] = AWS_domain.apply(lambda r: self.get_V(r), axis=1)
        AWS_domain = AWS_domain.dropna(axis=0, how='any')
        # resample data to even timestep (30min) e.g. from 12:28 or 12:31 to 12:30
        even_timestep = pd.date_range(
            start=timestart, end=analysis_time,  freq='30Min')
        UU = pd.DataFrame(even_timestep.values)
        UU = UU.rename(index=str, columns={0: 'time_stamp'})
        VV = UU
        TT = UU
        # reshape u,v,T to separate 2D arrays: (time,station)
        for i, ID in enumerate(AWS_ID):
            # U
            Ui = AWS_domain[AWS_domain['ID_num'] == ID][['time_stamp', 'u']]
            # should we set missingValue #threshold??
            Ui = Ui.set_index('time_stamp').resample('30Min').mean()
            Ui = Ui.interpolate(method='time')  # interpolate missingValue now
            Ui['time_stamp'] = Ui.index
            Ui = Ui.rename(index=str, columns={
                           'time_stamp': 'time_stamp', 'u': str(ID)})
            if Ui.shape[0] != 0:
                UU = pd.merge(UU, Ui)
            # V
            Vi = AWS_domain[AWS_domain['ID_num'] == ID][['time_stamp', 'v']]
            Vi = Vi.set_index('time_stamp').resample('30Min').mean()
            Vi = Vi.interpolate(method='time')
            Vi['time_stamp'] = Vi.index
            Vi = Vi.rename(index=str, columns={
                           'time_stamp': 'time_stamp', 'v': str(ID)})
            if Vi.shape[0] != 0:
                VV = pd.merge(VV, Vi)
            # T
            Ti = AWS_domain[AWS_domain['ID_num'] == ID][['time_stamp', 'T_DB']]
            Ti = Ti.set_index('time_stamp').resample('30Min').mean()
            Ti = Ti.interpolate(method='time')
            Ti['time_stamp'] = Ti.index
            Ti = Ti.rename(index=str, columns={
                           'time_stamp': 'time_stamp', 'T_DB': str(ID)})
            if Ti.shape[0] != 0:
                TT = pd.merge(TT, Ti)
        # perform EOF and reconstruction
        Date_time = UU.values[:, 0]
        U_recon = self.EOF_core(UU.values[:, 1:].astype('float'))
        V_recon = self.EOF_core(VV.values[:, 1:].astype('float'))
        T_recon = self.EOF_core(TT.values[:, 1:].astype('float'))
        Wspd_recon = (U_recon**2+V_recon**2)**0.5
        r2d = 45.0/math.atan(1.0)
        Wdir_recon = np.arctan2(U_recon, V_recon) * r2d + 180
        Wspd_orig = (UU.values[:, 1:].astype('float') **
                     2+VV.values[:, 1:].astype('float')**2)**0.5
        Wdir_orig = np.arctan2(UU.values[:, 1:].astype(
            'float'), VV.values[:, 1:].astype('float')) * r2d + 180
        df_out = pd.DataFrame({'ID_num': [], 'ID_name[6]': [], 'Lat': [], 'Lon': [], 'Height (m)': [],
                             'time_stamp': [], 'Wdir': [], 'Wspd': [], 'T_DB': [], 'DP': []})
        output_index = UU.values[:, 0].tolist().index(analysis_time-timedelta(hours=5))
        for i in range(U_recon.shape[1]):
            aws_info = BoMData[BoMData['ID_num']== int(UU.columns[i+1])].iloc[0]
            df_i = pd.DataFrame({'ID_num': UU.columns[i+1],
                                 'ID_name[6]': aws_info['ID_name[6]'],
                                 'Lat': aws_info.Lat,
                                 'Lon': aws_info.Lon,
                                 'Height (m)': aws_info['Height (m)'],
                                 'time_stamp': UU.values[output_index:, 0],
                                 'Wdir': Wdir_recon[output_index:, i],
                                 'Wspd': Wspd_recon[output_index:, i],
                                 'T_DB': T_recon[output_index:, i],
                                 'DP': -9999})    # skip dew point, won't use it in WRFDA anyway (need pressure data).
            df_out = pd.concat([df_out, df_i])
        df_out['T_DB'] = df_out['T_DB']+273.15
        df_out = df_out[['ID_num', 'ID_name[6]', 'Lat', 'Lon', 'Height (m)', 
                         'time_stamp', 'Wdir', 'Wspd', 'T_DB', 'DP']]
        return df_out

    def EOF_core(self, T_value):
        # omitting core code for EOF procedure
        pass

        
# dispatch system        
class MainControl(Model):
    def __init__(self):
        Model.__init__(self)
        # settings in control file, should be like this:
        '''
        self.run_27_manually = False # if pause before the last run.
        self.total_hours=30
        self.DAdomains=[1,2,3] 
        self.do_EOF=True
        self.useRadar=True
        self.useAWS=True
        self.radar_prep_method=0
        self.RadarDA_inter=5
        '''
        self.run_dir = self.env['RUN_DIR']
        (self.run_27_manually, self.total_hours, self.DAdomains, self.useRadar, self.useAWS,
            self.do_EOF, self.radar_prep_method, self.RadarDA_inter, _, _, _, _, _, self.restart_DA, self.restart_time) = DAsimulation_info()
        self.gfs_ana_time = hk.get_date_from_file(
            self.run_dir+'current_forecast')
        hk.write_log(self.run_dir, 'Prepare to run WRFDA, analysis time is: ' +
                     self.gfs_ana_time.strftime('%Y-%m-%d_%H:%M:%S'))
        print('Prepare to run WRFDA, analysis time is: ' +
              self.gfs_ana_time.strftime('%Y-%m-%d_%H:%M:%S'))

    def time_forward_str(self, nhours):
        next_time = self.gfs_ana_time + timedelta(hours=nhours)
        return next_time.strftime('%Y%m%d%H%M')

    def run_DA(self, current_time):
        if self.useRadar:
            radar_prep1 = RadarPrep(current_time, self.radar_prep_method)
            radar_prep1.run_prep()
        rstartimes = [0, 0, 0]
        timestring = current_time.strftime('%Y%m%d%H%M')
        for domain in self.DAdomains:  
            hk.write_log(self.run_dir, 'Running DA for domain_'+str(domain) +
                         ' '+current_time.strftime('%Y-%m-%d_%H:%M:%S')+'.')
            try:
                os.remove(self.env['WRFDA_domains'] +
                          'wrfvar_d0' + str(domain) + '/ob.ascii')
            except FileNotFoundError:
                pass
            if self.useAWS:
                if (current_time.strftime('%M') == '30') or (current_time.strftime('%M') == '00'):
                    obsproc1 = ObsProc(current_time, domain, self.do_EOF)
                    obsproc1.run_obsproc()
                    # all_files=glob.glob(self.env['obs_dir']+'/*')
                    all_files = next(os.walk(self.env['obs_dir']))[2]
                    save_list = ['obs_6hours.csv', 'write_little_r.exe', 'obserr.txt', 'AWSInfo_trim.csv', 'write_little_r.f90',
                                 'BoM_AWS_IdLatLonElev_d01.csv', 'BoM_AWS_IdLatLonElev_d03.csv', 'obsproc.exe']
                    output_files = [
                        x.split('/')[-1] for x in all_files if not(x.split('/')[-1] in save_list)]
                    subprocess.run('mkdir results_'+str(timestring),
                                   cwd=self.env['obs_dir'], shell=True)
                    for item in output_files:
                        cp_command1 = subprocess.Popen(
                            'mv -- '+item+' ./results_'+str(timestring), cwd=self.env['obs_dir'], shell=True)
                        cp_command1.wait()
            if current_time == self.gfs_ana_time:
                # initial, fg is wrfinput from WPS
                fg_file = 'wrfinput_d0'+str(domain)
                rstartimes[domain-1] = self.gfs_ana_time
            else:
                # cycle, fg is wrf_3dvar_input generated by WRF and updated by lowerBC
                # work out wrf_3dvar_input timestamp (second could be random: wrf_3dvar_input_d03_2018-05-03_18:36:06)
                files = []
                while not files:
                    files = glob.glob(
                        self.env['WRF_domains']+'wrf_3dvar_input_d0'+str(domain)+'*')
                    sleep(10)
                tmstps = [pd.to_datetime(
                    x[-19:], format='%Y-%m-%d_%H:%M:%S') for x in files]
                tmdiffsec = [abs(x-current_time).seconds for x in tmstps]
                ind = (np.array(tmdiffsec) == min(tmdiffsec)) & (
                    np.array(tmdiffsec) <= 180)
                right_file = np.array(files)[ind]
                if right_file.shape[0] == 0:
                    print('file error in DA: '+'wrf_3dvar_input_d0'+str(domain))
                    print('looking for: ' +
                          current_time.strftime('%Y-%m-%d_%H:%M:%S'))
                    # exit next line..
                rstartimes[domain-1] = np.array(tmstps)[ind][0]
                fg_file = right_file[0].split('/')[-1]
            da_wrfvar1 = DAWRFvar(timestring, fg_file, domain)
            da_wrfvar1.run_model()
        # check progress
        completed = False
        time_wait = 0
        while not completed:
            sleep(60)
            time_wait = time_wait+1
            results = []
            for domain in self.DAdomains:
                da_rsl_file = self.env['wrfvar_dir'] + \
                    '_d0'+str(domain)+'/rsl.out.0000'
                if os.path.isfile(da_rsl_file):
                    # job started for this domain, check log to determine iresult.
                    iresult = check_DA_prog(da_rsl_file)
                else:   
                    # job hasn't started for this domain, set iresult to False.
                    iresult = False
                results.append(iresult)
            if np.array(results).all():
                completed = True
            msg = '{} : {}'.format('minutes passed', time_wait)
            hk.write_log(self.run_dir, msg)
            hk.write_status_log(self.run_dir, msg)
        msg = 'WRFDA finsihed'
        hk.write_log(self.run_dir, msg)
        # all_files=glob.glob(self.env['WRFDA_domains']+'wrfvar_d01/*')
        all_files = next(os.walk(self.env['WRFDA_domains']+'wrfvar_d01'))[2]
        save_list = ['da_wrfvar.exe', 'LANDUSE.TBL',
                     'wrfvar_output', 'wrfvar_output_d01']
        output_files = [x.split('/')[-1]
                        for x in all_files if not(x.split('/')[-1] in save_list)]
        for domain in self.DAdomains:
            cp_command = subprocess.Popen('mv ./wrfvar_output ./wrfvar_output_d0'+str(
                domain), cwd=self.env['WRFDA_domains']+'wrfvar_d0'+str(domain), shell=True)
            cp_command.wait()
            subprocess.run('mkdir results_'+str(timestring),
                           cwd=self.env['WRFDA_domains']+'wrfvar_d0'+str(domain), shell=True)
            for item in output_files:
                cp_command1 = subprocess.Popen('mv -- '+item+' ./results_'+str(
                    timestring), cwd=self.env['WRFDA_domains']+'wrfvar_d0'+str(domain), shell=True)
                cp_command1.wait()
        if current_time != self.gfs_ana_time:
            if (np.array(rstartimes) == 0).any():
                print('error getting wrf_3dvar_input timestamps!')
                os._exit(0)
        return rstartimes

    def run_WRF(self, after_DA, time_start, time_end, if_3dvar):
        timestring = time_start[0].strftime('%Y-%m-%d_%H:%M:%S')
        run_minutes = round((time_end-time_start[0]).total_seconds()/60)
        hk.write_log(self.run_dir, 'Running WRF from '+timestring +
                     ', to '+time_end.strftime('%Y-%m-%d_%H:%M:%S'))
        WRF1 = WRF(time_start, time_end, if_3dvar, self.DAdomains)
        WRF1.create_namelist()
        hk.write_tslist()
        if after_DA:
            WRF1.prep_file()
        if (if_3dvar == False) and (self.run_27_manually == True):
            print('pause before running 27h WRF')
            os._exit(0)
        WRF1.run_model()
        # move all outputs away
        output_files = ['wrfout*', '*.log', 'rsl.*', '*.TH',
                        '*.TS', '*.UU', '*.VV', '*.PH', '*.QV', 'dbz_d0*']
        subprocess.run('mkdir results_'+str(timestring)+'_' +
                       str(run_minutes), cwd=self.env['WRF_domains'], shell=True)
        for item in output_files:
            cp_command1 = subprocess.Popen('mv -- '+item+' ./results_'+str(
                timestring)+'_'+str(run_minutes), cwd=self.env['WRF_domains'], shell=True)
            cp_command1.wait()

    def check_obs(self):
        nobs = np.array([0, 0, 0, 0, 0])
        for j in range(5):
            time_j = self.gfs_ana_time+timedelta(hours=j)
            time_beg = time_j-timedelta(minutes=15)
            time_end = time_j+timedelta(minutes=15)
            stamps = pd.date_range(start=time_beg, end=time_end,  freq='15Min')
            timestr5 = ['']*stamps.size
            for i in range(stamps.size):
                stpstr = str(stamps[i])  
                timenogap = stpstr[0:4]+stpstr[5:7] + \
                    stpstr[8:10]+stpstr[11:13]+stpstr[14:16]
                timestr5[i] = timenogap
            dataDir = self.env['AWS_dir']
            filelist = []
            for i in range(stamps.size):
                filei = glob.glob(dataDir+'*'+timestr5[i]+'*.axf')
                try:
                    filelist.append(filei[0])
                except:
                    continue
            filelist.sort()
            nobs[j] = len(filelist)
        obs_stat = nobs > 0    
        return obs_stat

    def check_radar(self, freqc):
        stamps = pd.date_range(
            start=self.gfs_ana_time, end=self.gfs_ana_time+timedelta(hours=4),  freq=f'{6*freqc}Min')
        data_dir = self.env['Radar_dir']
        obstime = []
        for i in range(stamps.shape[0]):
            tagettime = stamps[i]
            dateNtime = tagettime.strftime('%Y%m%d_%H%M').split('_')
            radar_file = glob.glob(
                data_dir+dateNtime[0]+'/'+dateNtime[1]+'*.h5')
            if len(radar_file) == 1:
                obstime.append(stamps[i])
        return obstime

    def init_run(self, first_run):
        if first_run != self.gfs_ana_time:
            self.run_WRF(False, [self.gfs_ana_time]*3, first_run, True)

    def cycle_run(self, hour_start, hour_end):    # DA-BC_lat-WRF-BC_low
        if self.restart_DA:
            if hour_start < self.restart_time:
                print('in RESTART mode, skipping timestamp: ' +
                      hour_start.strftime('%Y-%m-%d_%H:%M:%S'))
                return  # should finish WRF for restart_time.
        rstartimes = self.run_DA(hour_start)
        if hour_start.strftime('%M') == '00':
            update_lat = UpdateBC(hour_start, 'lateral', 1)
            update_lat.execute_all()
        self.run_WRF(True, rstartimes, hour_end, True)
        if hour_end.strftime('%M') == '00':
            for domain in self.DAdomains:
                update_low = UpdateBC(hour_end, 'lower', domain)
                update_low.execute_all()

    def final_run(self, hour_start):    # DA-BC_lat-WRF
        rstartimes = self.run_DA(hour_start)
        if hour_start.strftime('%M') == '00':
            update_lat = UpdateBC(hour_start, 'lateral', 1)
            update_lat.execute_all()
        print('ready for the last run')
        hk.write_log(self.run_dir, 'ready for the last run')
        self.run_WRF(True, rstartimes, self.gfs_ana_time +
                     timedelta(hours=self.total_hours), False)

    def EOF(self):
        hk.write_log(self.run_dir, 'Doing EOFQC')
        last_hour = self.gfs_ana_time+timedelta(hours=4)
        EOF_QC1 = EOFQC()
        df_out = EOF_QC1.EOF_func(last_hour)
        df_out.to_csv(self.env['obs_dir']+'obs_6hours.csv',
                      sep=',', header=True, index=False)

    def main_func(self):
        # cp (don't overwrite) original wrfinput and wrfbdy to ./backup/, if DA fails restore them in auto.py
        subprocess.run('cp -n ./wrfbdy_d01 ./backup/',
                       cwd=self.env['WRF_domains'], shell=True)
        subprocess.run('cp -n ./wrfinput_d* ./backup/',
                       cwd=self.env['WRF_domains'], shell=True)
        DA_times = self.check_radar(self.RadarDA_inter)
        runtime_start = self.gfs_ana_time
        runtime_end = self.gfs_ana_time+timedelta(hours=self.total_hours)
        isobs = len(DA_times) > 0
        DAresult = True
        if isobs:
            try: # main loop:
                hk.write_log(self.run_dir, 'Conducting DA ' +
                             str(len(DA_times))+' times')
                print('Conducting DA '+str(len(DA_times))+' times')
                if self.useAWS and self.do_EOF:
                    self.EOF()
                self.init_run(DA_times[0])
                if (len(DA_times)-1) > 0:
                    for i in range(len(DA_times)-1):
                        self.cycle_run(DA_times[i], DA_times[i+1])        
                self.final_run(DA_times[-1])
            except:
                hk.write_log(self.run_dir, 'Runtime error, exit now.')
                print('Runtime error, exit now.')     
                DAresult = False        
        else:
            hk.write_log(self.run_dir, 'No new files, retry later.')
            print('No new files, retry later.')
            DAresult = False
        return DAresult     # in auto.py:  if DAresult==False : WRF_noDA()


if __name__ == "__main__":
    print('in __main__')
    DA_run = MainControl()
    DA_run.main_func()
    exit()
