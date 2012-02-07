# WEMD run control routines

from __future__ import division, print_function; __metaclass__ = type

import logging
log = logging.getLogger('wemd.rc')

import os, sys, argparse, errno
import wemd
from wemd.util.config_dict import ConfigDict
from wemd.util import extloader


def lazy_loaded(backing_name, loader, docstring = None):
    def getter(self):
        obj = getattr(self, backing_name, None)
        if obj is None:
            obj = loader()
            setattr(self,backing_name,obj)
        return obj
    def setter(self, val):
        setattr(self,backing_name,val)
    def deleter(self):
        delattr(self,backing_name)
        setattr(self,backing_name,None)
    return property(getter, setter, deleter, docstring)

class _WEMDRC:
    '''A class, an instance of which is accessible as ``wemd.rc``, to handle global issues for WEMD code,
    such as loading modules and plugins, writing output based on verbosity level, adding default command line options,
    and so on.'''
    
    # Runtime config file management
    ENV_RUNTIME_CONFIG  = 'WEMDRC'
    RC_DEFAULT_FILENAME = 'wemd.cfg'
    
    DEFAULT_WORK_MANAGER = 'zmq'
        
    def __init__(self):        
        self.verbosity = None
        self.rcfile = os.environ.get(self.ENV_RUNTIME_CONFIG) or self.RC_DEFAULT_FILENAME

        self.config = ConfigDict()
        self.process_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    
        self._work_manager_args_added = False
        self.work_manager = None
        self.system = None
        
    def add_args(self, parser):
        group = parser.add_argument_group('general options')
        group.add_argument('-r', '--rcfile', metavar='RCFILE', dest='rcfile',
                            default=(os.environ.get(self.ENV_RUNTIME_CONFIG) or self.RC_DEFAULT_FILENAME),
                            help='use RCFILE as the WEMD run-time configuration file (default: %(default)s)')
        
        egroup = group.add_mutually_exclusive_group()
        egroup.add_argument('--quiet', dest='verbosity', action='store_const', const='quiet',
                             help='emit only essential information')
        egroup.add_argument('--verbose', dest='verbosity', action='store_const', const='verbose',
                             help='emit extra information')
        egroup.add_argument('--debug', dest='verbosity', action='store_const', const='debug',
                            help='enable extra checks and emit copious information')
        
        group.add_argument('--version', action='version', version='WEMD version %s' % wemd.version)
        
    def add_work_manager_args(self, parser):
        self._work_manager_args_added = True
        group = parser.add_argument_group('work manager options')
        group.add_argument('--work-manager', dest='work_manager_name',
                            help='''use the given work manager to distribute work among processors (serial, threads, processes, 
                            zmq, or name a Python class as ``module.name``; default: %(default)s)''')
        group.add_argument('--help-work-manager', dest='do_work_manager_help', action='store_true',
                            help='display help specific to the given work manager')
                
    
    @property
    def verbose_mode(self):
        return (self.verbosity == 'verbose')
    
    @property
    def debug_mode(self):
        return (self.verbosity == 'debug')
    
    @property
    def quiet_mode(self):
        return (self.verbosity == 'quiet')
                            
    def process_args(self, args, aux_args=[], config_required = True):
        self.cmdline_args = args
        self.verbosity = args.verbosity
        if self._work_manager_args_added:
            self.work_manager_name = args.work_manager_name 
        
        if args.rcfile:
            self.rcfile = args.rcfile
        
        try:
            self.read_config()
        except IOError as e:
            if e.errno == errno.ENOENT and not config_required:
                pass
            else:
                raise
        self.config_logging()
        self.config.update_from_object(args)
        
        if self._work_manager_args_added:
            work_manager = self.get_work_manager()
            aux_args = work_manager.parse_aux_args(aux_args, do_help=args.do_work_manager_help)
            if aux_args:
                sys.stderr.write('unexpected command line argument(s) encountered: {}\n'.format(aux_args))
                sys.exit(os.EX_USAGE)
                    
    def read_config(self, filename = None):
        if filename:
            self.rcfile = filename

        if 'WEMD_SIM_ROOT' not in os.environ:
            sys.stderr.write('  -- WARNING  -- setting $WEMD_SIM_ROOT to current directory ({})\n'.format(os.getcwd()))
            os.environ['WEMD_SIM_ROOT'] = os.getcwd()
                                    
        self.config.read_config_file(self.rcfile) 
                    
    def config_logging(self):
        import logging.config
        logging_config = {'version': 1, 'incremental': False,
                          'formatters': {'standard': {'format': '  -- %(levelname)-8s -- %(message)s'},
                                         'debug':    {'format': '''\
          -- %(levelname)-8s %(asctime)24s PID %(process)-12d TID %(thread)-20d 
             %(pathname)s:%(lineno)d [%(funcName)s()] 
               %(message)s'''}},
                          'handlers': {'console': {'class': 'logging.StreamHandler',
                                                   'stream': 'ext://sys.stdout',
                                                   'formatter': 'standard'}},
                          'loggers': {'wemd': {'handlers': ['console'], 'propagate': False},
                                      'wemdtools': {'handlers': ['console'], 'propagate': False},
                                      'wemdext': {'handlers': ['console'], 'propagate': False}},
                          'root': {'handlers': ['console']}}
        
        logging_config['loggers'][self.process_name] = {'handlers': ['console'], 'propagate': False}
            
        if self.verbosity == 'debug':
            logging_config['root']['level'] = 'DEBUG'
            logging_config['handlers']['console']['formatter'] = 'debug'
        elif self.verbosity == 'verbose':
            logging_config['root']['level'] = 'INFO'
        else:
            logging_config['root']['level'] = 'WARNING'

        logging.config.dictConfig(logging_config)
        logging_config['incremental'] = True
        
    def pstatus(self, *args, **kwargs):
        fileobj = kwargs.get('file', sys.stdout)
        if kwargs.get('termonly', False) and not fileobj.isatty():
            return
        if self.verbosity != 'quiet':
            print(*args, **kwargs)
        
    def pflush(self):
        sys.stdout.flush()
        sys.stderr.flush()
        
    def get_sim_manager(self):
        drivername = self.config.get('drivers.sim_manager', 'default')
        if drivername.lower() == 'default':
            from wemd.sim_manager import WESimManager
            return WESimManager()
        else:
            pathinfo = self.config.get_pathlist('drivers.module_path')
            return extloader.get_object(drivername,pathinfo)()
        
    def get_data_manager(self):
        drivername = self.config.get('drivers.data_manager', 'hdf5')
        if drivername.lower() in ('hdf5', 'default'):
            data_manager = wemd.data_manager.WEMDDataManager()
        else:
            pathinfo = self.config.get_pathlist('drivers.module_path', default=None)
            data_manager = extloader.get_object(drivername, pathinfo)()
        log.debug('loaded data manager: {!r}'.format(data_manager))
        return data_manager

    def get_we_driver(self):
        drivername = self.config.get('drivers.we_driver', 'default')
        if drivername.lower() == 'default':
            we_driver = wemd.we_driver.WEMDWEDriver()
        else:
            pathinfo = self.config.get_pathlist('drivers.module_path', default=None)
            we_driver = extloader.get_object(drivername, pathinfo)()
        log.debug('loaded WE algorithm driver: {!r}'.format(we_driver))
        return we_driver

    def get_work_manager(self):
        if self.work_manager:
            return self.work_manager
        else:
            drivername = self.config.get('args.work_manager_name')
            if not drivername:
                drivername = self.config.get('drivers.work_manager', 'default')
            ldrivername = drivername.lower()
            if ldrivername == 'default':
                ldrivername = self.DEFAULT_WORK_MANAGER
                
            if ldrivername == 'serial':
                import wemd.work_managers.serial
                work_manager = wemd.work_managers.serial.SerialWorkManager()
            elif ldrivername == 'threads':
                import wemd.work_managers.threads
                work_manager = wemd.work_managers.threads.ThreadsWorkManager()
            elif ldrivername == 'processes':
                import wemd.work_managers.processes
                work_manager = wemd.work_managers.processes.ProcessWorkManager()
            elif ldrivername in ('zmq', 'zeromq', '0mq'):
                import wemd.work_managers.zeromq
                work_manager = wemd.work_managers.zeromq.ZMQWorkManager()
            elif '.' in ldrivername:
                pathinfo = self.config.get_pathlist('drivers.module_path', default=None)
                work_manager = extloader.get_object(drivername, pathinfo)()
            else:
                raise ValueError('unknown work manager {!r}'.format(drivername))
            log.debug('loaded work manager: {!r}'.format(work_manager))
            self.work_manager = work_manager
            return work_manager
    
    def get_propagator(self):
        drivername = self.config.require('drivers.propagator')
        if drivername.lower() == 'executable':
            import wemd.propagators.executable
            propagator = wemd.propagators.executable.ExecutablePropagator()
        else:
            pathinfo = self.config.get_pathlist('drivers.module_path', default=None)
            propagator = extloader.get_object(drivername, pathinfo)()
        log.debug('loaded propagator {!r}'.format(propagator))
        return propagator
    
    def get_system_driver(self):
        if self.system is None:
            sysdrivername = self.config.require('system.system_driver')
            log.info('loading system driver %r' % sysdrivername)
            pathinfo = self.config.get_pathlist('system.module_path', default=None)
            try:        
                system = extloader.get_object(sysdrivername, pathinfo)()
            except ImportError:
                extra_path = os.environ.get('WEMD_SIM_ROOT', '.')
                try:
                    system = extloader.get_object(sysdrivername, [extra_path])()
                except ImportError:
                    raise ImportError('could not load system driver')
                else:
                    log.warning('using system driver from {!r}'.format(extra_path))
            log.debug('loaded system driver {!r}'.format(system))
            
            log.debug('initializing system driver')
            system.initialize()
            self.system = system
        return self.system