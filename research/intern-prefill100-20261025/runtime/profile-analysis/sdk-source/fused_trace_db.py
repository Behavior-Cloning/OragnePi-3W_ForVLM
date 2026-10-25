from collections import OrderedDict
from tools.device_profiler.utils import list_add


class FusedTraceDB:
    def __init__(self, nb_net, uid):
        self.__nb_net = nb_net
        self.__uid = uid
        self.__all_org_shared_uids = list()
        self.__all_shared_fused_uids = list()
        self.__all_shared_node_ids = list()
        self.__input_shapes = None
        self.__output_shapes = None

    @property
    def all_shared_org_uids(self):
        return self.__all_org_shared_uids

    @all_shared_org_uids.setter
    def all_shared_org_uids(self, value):
        self.__all_org_shared_uids = value

    @property
    def all_shared_node_ids(self):
        return self.__all_shared_node_ids

    @all_shared_node_ids.setter
    def all_shared_node_ids(self, value):
        self.__all_shared_node_ids = value

    @property
    def all_shared_fused_uids(self):
        return self.__all_shared_fused_uids

    @all_shared_fused_uids.setter
    def all_shared_fused_uids(self, value):
        self.__all_shared_fused_uids = value

    @property
    def input_shapes(self):
        return self.__input_shapes

    @input_shapes.setter
    def input_shapes(self, value):
        self.__input_shapes = value

    @property
    def output_shapes(self):
        return self.__output_shapes

    @output_shapes.setter
    def output_shapes(self, value):
        self.__output_shapes = value

    def add_attrs(self, trace_dict, fused_layers, trace_item):
        self.input_shapes = trace_item.get('input_shapes')
        self.output_shapes = trace_item.get('output_shapes')
        org_fused_ids = trace_item.get('fused_json_uids')
        # 1. get all shared fused uids.
        self.all_shared_fused_uids = self.__get_all_shared_fused_uids(fused_layers, org_fused_ids)

        # 2. get all shared original json uids by all fused uids.
        self.all_shared_org_uids, self.all_shared_node_ids =\
            self.__get_shared_ids_by_fused_uid(trace_dict, fused_layers, self.all_shared_fused_uids)

    def __get_shared_ids_by_fused_uid(self, trace_dict, fused_layers, all_shared_fused_uids):
        all_org_shared_uids = []
        all_shared_node_ids = []
        for fused_uid in all_shared_fused_uids:
            share_uids = []
            for item in trace_dict['Maps']:
                fused_json_ids = item.get('fused_json_uids')
                if fused_uid in fused_json_ids:
                    share_uids = item.get('share_uids')
                    break
            all_org_shared_uids.extend(share_uids)
            all_shared_node_ids.extend(fused_layers[fused_uid].nb_ids)
        all_org_shared_uids = list(set(all_org_shared_uids))
        all_shared_node_ids = list(set(all_shared_node_ids))
        return all_org_shared_uids, all_shared_node_ids

    def __get_all_shared_fused_uids(self, fused_layers, org_fused_ids):
        all_shared_fused_uids = []
        for fused_uid in org_fused_ids:
            shared_uids = self.__get_shared_fused_uids(fused_layers, fused_uid)
            all_shared_fused_uids.extend(shared_uids)
        all_shared_fused_uids = list(set(all_shared_fused_uids))
        return all_shared_fused_uids

    def __get_shared_fused_uids(self, fused_layers, fused_uid):
        all_shared_fused_uids = []
        for uid, l in fused_layers.items():
            all_shared_fused_uids = l.get_share_uids()
            if fused_uid in all_shared_fused_uids:
                break
        return all_shared_fused_uids

    def serialize(self, nodes_info):
        cycle_count = []
        read_bw = []
        write_bw = []
        mac = []
        for id_ in self.all_shared_node_ids:
            if not self.__nb_net.is_in_nb_net(id_):
                continue
            node = nodes_info[id_]
            cycle_count = list_add(cycle_count, node.cycle)
            read_bw = list_add(read_bw, node.read_bw)
            write_bw = list_add(write_bw, node.write_bw)
            mac = list_add(mac, node.mac)
        tab = OrderedDict()
        tab['cycle'] = "{}".format(cycle_count)
        tab['readbw'] = "{}".format(read_bw)
        tab['writebw'] = "{}".format(write_bw)
        tab['mac'] = "{}".format(mac)
        tab['share_uids'] = "{}".format(self.__all_org_shared_uids)
        return tab
