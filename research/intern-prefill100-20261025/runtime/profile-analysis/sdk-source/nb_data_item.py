from collections import OrderedDict
from struct import Struct

__all__ = ['DataItem']

def get_bit_value(buf, mask, shift_right = None):
    val = buf & mask
    if shift_right is not None:
        val = val >> shift_right
    return val

class ItemSegBase(object):
    def __init__(self, segment):
        self._buf = segment
        self.__param = OrderedDict()

    @property
    def param(self):
        return self.__param

    def parse(self):
        raise NotImplementedError()

# SEL HEAD
class ItemSegHEAD(ItemSegBase):
    def __init__(self, segment):
        super(ItemSegHEAD, self).__init__(segment)
        self.__struct = Struct('@QQ')
        self.parse()

    def parse(self):
        values = self.__struct.unpack(self._buf)
        self.param['NodeId'] = values[0]
        self.param['CoreID'] = values[1]

# SEL 0
class ItemSeg0(ItemSegBase):
    def __init__(self, segment):
        super(ItemSeg0, self).__init__(segment)
        self.__struct = Struct('@HHIII')
        self.parse()

    def parse(self):
        values = self.__struct.unpack(self._buf)
        self.param['NN_LAYER_ID'] = values[0]
        self.param['NN_LAYER_ID_OVFL'] = get_bit_value(values[1], 0x8000, 15)
        self.param['NN_INSTR_INFO'] = values[2]
        self.param['NN_TOTAL_BUSY_CYCLE'] = values[3]
        self.param['NN_TOTAL_BUSY_CYCLE_OVFL'] = get_bit_value(values[4], 0x80000000, 31)

# SEL 1
class ItemSeg1(ItemSegBase):
    def __init__(self, segment):
        super(ItemSeg1, self).__init__(segment)
        self.__struct = Struct('@IIII')
        self.parse()

    def parse(self):
        values = self.__struct.unpack(self._buf)
        self.param['NN_TOTAL_READ_CYCLE_DDR'] = values[0]
        self.param['NN_TOTAL_READ_CYCLE_DDR_OVFL'] = get_bit_value(values[1], 0x80000000, 31)
        self.param['NN_TOTAL_READ_VALID_BW_DDR'] = values[2]
        self.param['NN_TOTAL_READ_VALID_BW_DDR_OVFL'] = get_bit_value(values[3], 0x80000000, 31)

# SEL 2
class ItemSeg2(ItemSegBase):
    def __init__(self, segment):
        super(ItemSeg2, self).__init__(segment)
        self.__struct = Struct('@IIII')
        self.parse()

    def parse(self):
        values = self.__struct.unpack(self._buf)
        self.param['NN_TOTAL_WRITE_CYCLE_DDR'] = values[0]
        self.param['NN_TOTAL_WRITE_CYCLE_DDR_OVFL'] = get_bit_value(values[1], 0x80000000, 31)
        self.param['NN_TOTAL_WRITE_VALID_BW_DDR'] = values[2]
        self.param['NN_TOTAL_WRITE_VALID_BW_DDR_OVFL'] = get_bit_value(values[3], 0x80000000, 31)

# SEL 3
class ItemSeg3(ItemSegBase):
    def __init__(self, segment):
        super(ItemSeg3, self).__init__(segment)
        self.__struct = Struct('@IIII')
        self.parse()

    def parse(self):
        values = self.__struct.unpack(self._buf)
        self.param['NN_TOTAL_READ_CYCLE_SRAM'] = values[0]
        self.param['NN_TOTAL_READ_CYCLE_SRAM_OVFL'] = get_bit_value(values[1], 0x80000000, 31)
        self.param['NN_TOTAL_WRITE_CYCLE_SRAM'] = values[2]
        self.param['NN_TOTAL_WRITE_CYCLE_SRAM_OVFL'] = get_bit_value(values[3], 0x80000000, 31)

# SEL 4
class ItemSeg4(ItemSegBase):
    def __init__(self, segment):
        super(ItemSeg4, self).__init__(segment)
        self.__struct = Struct('@IIII')
        self.parse()

    def parse(self):
        values = self.__struct.unpack(self._buf)
        self.param['NN_TOTAL_MAC_CYCLE'] = values[0]
        self.param['NN_TOTAL_MAC_CYCLE_OVFL'] = get_bit_value(values[1], 0x80000000, 31)
        self.param['NN_TOTAL_MAC_COUNT'] = values[2]
        self.param['NN_TOTAL_MAC_COUNT_OVFL'] = get_bit_value(values[3], 0x80000000, 31)

# SEL 5
class ItemSeg5(ItemSegBase):
    def __init__(self, segment):
        super(ItemSeg5, self).__init__(segment)
        self.__struct = Struct('@IIII')
        self.parse()

    def parse(self):
        values = self.__struct.unpack(self._buf)
        self.param['NN_ZERO_COEF_SKIP_COUNT'] = values[0]
        self.param['NN_ZERO_COEF_SKIP_COUNT_OVFL'] = get_bit_value(values[1], 0x80000000, 31)
        self.param['NN_NON_ZERO_COEF_COUNT'] = values[2]
        self.param['NN_NON_ZERO_COEF_COUNT_OVFL'] = get_bit_value(values[3], 0x80000000, 31)

# SEL 15
class ItemSeg15(ItemSegBase):
    def __init__(self, segment):
        super(ItemSeg15, self).__init__(segment)
        self.__struct = Struct('@I12s')
        self.parse()

    def parse(self):
        values = self.__struct.unpack(self._buf)
        self.param['NN_TOTAL_IDLE_CYCLE_CORE_OVFL'] = values[0]

# SEL 20
class ItemSeg20(ItemSegBase):
    def __init__(self, segment):
        super(ItemSeg20, self).__init__(segment)
        self.__struct = Struct('@IIII')
        self.parse()

    def parse(self):
        values = self.__struct.unpack(self._buf)
        self.param['WAIT_FOR_IN_EVENT_CYCLE'] = values[0]
        self.param['WAIT_FOR_OUT_EVENT_CYCLE'] = values[1]
        self.param['WAIT_FOR_IN_EVENT_CYCLE_OVFL'] = get_bit_value(values[2], 0x80000000, 31)
        self.param['WAIT_FOR_OUT_EVENT_CYCLE_OVFL'] = get_bit_value(values[2], 0x40000000, 30)

# SEL 21
class ItemSeg21(ItemSegBase):
    def __init__(self, segment):
        super(ItemSeg21, self).__init__(segment)
        self.__struct = Struct('@IIII')
        self.parse()

    def parse(self):
        values = self.__struct.unpack(self._buf)
        self.param['WAIT_FOR_IN0_EVENT_CYCLE'] = values[0]
        self.param['WAIT_FOR_IN1_EVENT_CYCLE'] = values[1]
        self.param['WAIT_FOR_IN0_EVENT_CYCLE_OVFL'] = get_bit_value(values[2], 0x80000000, 31)
        self.param['WAIT_FOR_IN1_EVENT_CYCLE_OVFL'] = get_bit_value(values[2], 0x40000000, 30)

# SEL 22
class ItemSeg22(ItemSegBase):
    def __init__(self, segment):
        super(ItemSeg22, self).__init__(segment)
        self.__struct = Struct('@IIII')
        self.parse()

    def parse(self):
        values = self.__struct.unpack(self._buf)
        self.param['WAIT_FOR_OUT0_EVENT_CYCLE'] = values[0]
        self.param['WAIT_FOR_OUT1_EVENT_CYCLE'] = values[1]
        self.param['WAIT_FOR_OUT2_EVENT_CYCLE'] = values[2]
        self.param['WAIT_FOR_OUT0_EVENT_CYCLE_OVFL'] = get_bit_value(values[3], 0x80000000, 31)
        self.param['WAIT_FOR_OUT1_EVENT_CYCLE_OVFL'] = get_bit_value(values[3], 0x40000000, 30)
        self.param['WAIT_FOR_OUT2_EVENT_CYCLE_OVFL'] = get_bit_value(values[3], 0x20000000, 29)

# SEL 23
class ItemSeg23(ItemSegBase):
    def __init__(self, segment):
        super(ItemSeg23, self).__init__(segment)
        self.__struct = Struct('@IIQ')
        self.parse()

    def parse(self):
        values = self.__struct.unpack(self._buf)
        self.param['NN_IDLE_CYCLE'] = values[0]
        self.param['NN_IDLE_CYCLE_OVFL'] = get_bit_value(values[1], 0x80000000, 31)

class DataItem(object):
    def __init__(self, data_item):
        self.__buf = data_item
        self.__param = OrderedDict()
        self.__item_list = list()

        self.__seg_len = 16 # 128 bits or 16 bytes
        self.__start = 0
        self.__end = self.__seg_len

        self.__build_param()

    def __str__(self):
        return "NodeId={}, CoreId={}".format(self.node_id, self.core_id)

    @property
    def param(self):
        return self.__param

    @property
    def node_id(self):
        return self.__param['NodeId']

    @property
    def core_id(self):
        return self.__param['CoreID']

    def __build_seg(self, itemClass):
        item = itemClass(self.__buf[self.__start:self.__end])
        self.param.update(item.param)

        self.__start = self.__end
        self.__end += self.__seg_len
        return item

    def __build_param(self):
        # HW skipped some SEL X, they are reserved.
        self.__build_seg(ItemSegHEAD) # SEL HEAD, insert by viplite
        self.__build_seg(ItemSeg0)  # SEL 0
        self.__build_seg(ItemSeg1)  # SEL 1
        self.__build_seg(ItemSeg2)  # SEL 2
        self.__build_seg(ItemSeg3)  # SEL 3
        self.__build_seg(ItemSeg4)  # SEL 4
        self.__build_seg(ItemSeg5)  # SEL 5
        self.__build_seg(ItemSeg15)  # SEL 15
        self.__build_seg(ItemSeg20)  # SEL 20
        self.__build_seg(ItemSeg21)  # SEL 21
        self.__build_seg(ItemSeg22)  # SEL 22
        self.__build_seg(ItemSeg23)  # SEL 23
