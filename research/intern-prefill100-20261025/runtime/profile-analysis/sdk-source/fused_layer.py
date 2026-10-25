
__all__ = ['FusedLayer']

class FusedLayer(object):
    def __init__(self, uid):
        self.__uid = uid
        self.__share_ids = set()
        self.__nb_ids = set()

    def __str__(self):
        return str(self.__uid)

    @property
    def uid(self):
        return self.__uid

    @property
    def nb_ids(self):
        return list(self.__nb_ids)

    def get_share_uids(self):
        share = set()
        for trace_id in self.__share_ids:
            uid = int(trace_id.split('_')[0])
            share.add(uid)
        return list(share)

    def add_traces(self, trace):
        assert isinstance(trace, list)
        for t in trace:
            self.__share_ids.update(t.shared_ids)
            self.__nb_ids.update(t.get_profile_nodes_id())
