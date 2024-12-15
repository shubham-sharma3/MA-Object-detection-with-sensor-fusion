import sys

sys.path.append('osi3') #subfolder of generated python-code for the protobuf descriptors

#import osi classes
from osi3.osi_sensordata_pb2 import SensorData
import google.protobuf.text_format as text_format
from tqdm import tqdm

import struct

file_name = "C:\\Users\\Proteus\\Desktop\\RWTH\\Master_Thesis\\Kitti_dataset_writer\\label_data\\1234\\SimRecords_2024-04-01_23-04-27\\20240402T000123Z_sd_340_3120_350_.osi"

def parse_BBs_osi(file_name):
    print("Reading .osi file: {}".format(file_name))
    with open(file_name, 'rb') as file:
        #note: serialization always has header to indicate payload length, then the message data
        #bytes_buffer = sensorview.SerializeToString()
        #f.write(struct.pack("<L", len(bytes_buffer)))
        #f.write(bytes_buffer)

        # i.e. we first write the length of the message as little-endian long integer
        #    and the length of the buffer, then the message itself (repeatedly until the file is at end

        labels = []
        progress = tqdm()
        while (True):
            frame = []
            header_content = file.read(4)


            if header_content == b'':
                break

            #print (f"header: {int(header_content)}")

            payload_length = struct.unpack("<L", header_content)[0]
            # print(f"payload_length:{payload_length}")

            payload = file.read(payload_length)
            sensor_data = SensorData()
            sensor_data.ParseFromString(payload)
            #human-readable string representation
            text_proto = text_format.MessageToString(sensor_data)
            # print(text_proto)

            #data access
            cam_data = sensor_data.feature_data.camera_sensor[0]
            point_list = cam_data.point

            for detection in cam_data.detection:

                # print(f"detection for object id {detection.object_id.value}")
                # if (detection.shape_classification_vehicle):
                #     print("\\tvehicle")
                # elif (detection.shape_classification_pedestrian):
                #     print("\tpedestrian")
                # elif (detection.shape_classification_traffic_sign):
                #     print("\ttraffic_sign")
                # elif (detection.shape_classification_traffic_light):
                #     print("\ttraffic_light")
                # elif (detection.shape_classification_road_marking):
                #     print("\troad_marking")
                # else:
                #     print(f"\tunknown type:, dumping entire detection\n{detection}")
                #sensordata has arrays of moving_object, stationary_object, traffic_light, traffic_sign, road_marking,
                #the header.tracking_id.value allows cross-references into the detections for additional information

                #read index into point list and parse x/y pixel coordinates
                #bounding boxes always have 2 points, detection.image_shape_type is always "BOX"
                # bb_min = point_list[detection.first_point_index]
                # bb_max = point_list[detection.first_point_index + 1]
                #
                # print(f"\tbb_min x={bb_min.pixel_x}, y={bb_min.pixel_y}")
                # print(f"\tbb_min x={bb_max.pixel_x}, y={bb_max.pixel_y}")

                cls = [a for a in dir(detection) if a.startswith('shape_classification_') and getattr(detection, a) == True][0]
                if cls == 'shape_classification_road_marking': continue
                if cls == 'shape_classification_traffic_sign': continue
                box = [point_list[detection.first_point_index].pixel_x, point_list[detection.first_point_index].pixel_y,
                       point_list[detection.first_point_index + 1].pixel_x, point_list[detection.first_point_index + 1].pixel_y]
                # frame["classes"].append(cls)
                # frame["boxes"].append(box)
                frame.append({'class': cls, 'box': box})

            labels.append(frame)
            progress.update()
            # break #remove this to parse all messages (not just the first)

    ## Inspect classes of labels
    # cls_total = []
    # for label in labels:
    #     cls_total += label['classes']
    # print(list(set(cls_total)))

    print("finished reading file")
    img_size = [1920, 1280] #TODO: read this value from the labels or even the size of the images instead of giving it
    return labels, img_size

if __name__ == '__main__':
    parse_BBs_osi(file_name)