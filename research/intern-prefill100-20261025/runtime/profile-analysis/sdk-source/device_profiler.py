import copy
from collections import OrderedDict
from tools.device_profiler.nb_data import NBProfileData
from tools.device_profiler.node_trace import NodeTraceDB
from tools.device_profiler.fused_layer import FusedLayer
from tools.device_profiler.fused_trace_db import FusedTraceDB
from tools.device_profiler.utils import load_json, parse_lid, list_add
from tools.device_profiler.node_trace import TargetType


__all__ = ['DeviceProfiler']


class OrgNet(object):
    def __init__(self, model_file):
        self.__net = load_json(model_file)

    @property
    def layers(self):
        return self.__net['Layers']

    @property
    def meta_data(self):
        return self.__net['MetaData']

    @property
    def network_name(self):
        return self.__net['MetaData']['Name']


class NodeId:
    def __init__(self, node_id):
        self.__node_id = node_id
        self.__cycle = list()
        self.__read_bw = list()
        self.__write_bw = list()
        self.__mac = list()

    @property
    def node_id(self):
        return self.__node_id

    @property
    def cycle(self):
        return self.__cycle

    @property
    def read_bw(self):
        return self.__read_bw

    @property
    def write_bw(self):
        return self.__write_bw

    @property
    def mac(self):
        return self.__mac

    @cycle.setter
    def cycle(self, val):
        assert isinstance(val, list)
        self.__cycle = val

    @read_bw.setter
    def read_bw(self, val):
        assert isinstance(val, list)
        self.__read_bw = val

    @write_bw.setter
    def write_bw(self, val):
        assert isinstance(val, list)
        self.__write_bw = val

    @mac.setter
    def mac(self, val):
        assert isinstance(val, list)
        self.__mac = val

    def compute_profile_info(self, id, nb_net, trace_db):
        self.cycle = list_add(self.cycle, nb_net.get_cycle_count(id))
        self.read_bw = list_add(self.read_bw, nb_net.get_read_bw(id))
        self.write_bw = list_add(self.write_bw, nb_net.get_write_bw(id))
        self.mac = list_add(self.mac, nb_net.get_mac(id))

class DeviceProfiler(object):
    def __init__(self, model, fused_trace, node_trace, nb_data):
        self.__trace_dict = load_json(fused_trace)
        self.__fused_uids = self.__trace_dict['Fused_uids']
        self.__org_net = OrgNet(model)
        self.__trace = NodeTraceDB(node_trace)
        self.__nb = NBProfileData(nb_data)
        self.__nb_net = self.__nb.get_profile_net(0)
        self.__total_node_ids = set()
        self.__node_id_info = OrderedDict()  # node id with the cycle information
        self.__fused_layers = OrderedDict()
        self.__org_layers = OrderedDict()
        self.__trace_tbl = OrderedDict()
        self.__build_fused_layers()
        self.__calc_profile_info()

    def __build_fused_layers(self):
        for uid in self.__fused_uids:
            layer = FusedLayer(uid)
            trace_nodes = self.__trace.get_trace_nodes_by_uid(uid)
            if len(trace_nodes) > 0:
                layer.add_traces(trace_nodes)
            self.__total_node_ids = self.__total_node_ids.union(layer.nb_ids)
            self.__fused_layers[uid] = layer

    def __calc_profile_info(self):
        for node_id in self.__total_node_ids:
            node_id_struct = NodeId(node_id)
            if not self.__nb_net.is_in_nb_net(node_id):
                target = self.__trace.query_node_target(node_id)
                # Can't find the nn profile data in npd file, it is a error.
                if target == TargetType.NN:
                    assert False, "Error: Miss SDK node [{}] NN profile data".format(node_id)
                print("Warn: Node [{}] target is {}, no profile data".
                      format(node_id, TargetType.get_name(target)))
                continue
            node_id_struct.compute_profile_info(node_id, self.__nb_net, self.__trace)
            self.__node_id_info[node_id] = node_id_struct

    def __build_trace_tbl(self):
        for item in self.__trace_dict['Maps']:
            uid = item.get('uid').get('id')
            fused_db = FusedTraceDB(self.__nb_net, uid)
            fused_db.add_attrs(self.__trace_dict, self.__fused_layers, item)
            self.__trace_tbl[uid] = fused_db

    def __build_total_info(self):
        total_cycle = list()
        total_read_bw = list()
        total_write_bw = list()
        total_mac = list()
        for nb_id, node_id_struct in self.__node_id_info.items():
            if not self.__nb_net.is_in_nb_net(nb_id):
                continue
            total_cycle = list_add(total_cycle, node_id_struct.cycle)
            total_read_bw = list_add(total_read_bw, node_id_struct.read_bw)
            total_write_bw = list_add(total_write_bw, node_id_struct.write_bw)
            total_mac = list_add(total_mac, node_id_struct.mac)

        ret = OrderedDict()
        ret['cycle'] = "{}".format(total_cycle)
        ret['read_bw'] = "{}".format(total_read_bw)
        ret['write_bw'] = "{}".format(total_write_bw)
        ret['mac'] = "{}".format(total_mac)
        return ret

    def __build_org_dict(self):
        tab = OrderedDict([('MetaData', self.__org_net.meta_data)])
        tab['TotalProfiler'] = self.__build_total_info()
        tab['Layers'] = OrderedDict()
        for lid, obj in self.__org_net.layers.items():
            copy_obj = copy.deepcopy(obj)
            if 'parameters' in obj.keys():
                del copy_obj['parameters']
            tab['Layers'][lid] = copy_obj

            name, uid = parse_lid(lid)
            trace_tbl = self.__trace_tbl[uid]
            input_shapes = trace_tbl.input_shapes
            output_shapes = trace_tbl.output_shapes
            tab['Layers'][lid]['parameters'] = OrderedDict()
            tab['Layers'][lid]['parameters'] = trace_tbl.serialize(self.__node_id_info)
            if len(input_shapes) != 0:
                tab['Layers'][lid]['parameters']["input_shapes"] = input_shapes
            if len(output_shapes) != 0:
                tab['Layers'][lid]['parameters']["output_shapes"] = output_shapes
        return tab

    def __dump_profile(self, dict):
        import json
        filename = './{}.profile.json'.format(self.__org_net.network_name)
        print("Save profile to {}".format(filename))
        with open(filename, 'w') as f:
            json.dump(dict, f, indent=4)
        f.close()

    def profile(self):
        self.__calc_profile_info()
        self.__build_trace_tbl()
        dict = self.__build_org_dict()
        self.__dump_profile(dict)
        return dict
