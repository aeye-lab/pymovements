# Copyright (c) 2023-2025 The pymovements Project Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Test read from eyelink asc files."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


@pytest.mark.parametrize(
    ('kwargs', 'expected_frame'),
    [
        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_example.asc',
                'patterns': 'eyelink',
            },
            pl.from_dict(
                data={
                    'time': [
                        2154556, 2154557, 2154560, 2154564, 2154596, 2154598, 2154599, 2154695,
                        2154696, 2339227, 2339245, 2339246, 2339271, 2339272, 2339290, 2339291,
                    ],
                    'pupil': [
                        778.0, 778.0, 777.0, 778.0, 784.0, 784.0, 784.0, 798.0,
                        799.0, 619.0, 621.0, 622.0, 617.0, 617.0, 618.0, 618.0,
                    ],
                    'pixel': [
                        [138.1, 132.8], [138.2, 132.7], [137.9, 131.6], [138.1, 131.0],
                        [139.6, 132.1], [139.5, 131.9], [139.5, 131.8], [147.2, 134.4],
                        [147.3, 134.1], [673.2, 523.8], [629.0, 531.4], [629.9, 531.9],
                        [639.4, 531.9], [639.0, 531.9], [637.6, 531.4], [637.3, 531.2],
                    ],
                },
                schema={
                    'time': pl.Int64,
                    'pupil': pl.Float64,
                    'pixel': pl.List(pl.Float64),
                },
            ),
            id='eyelink_asc_mono_pattern_eyelink',
        ),
        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_example.asc',
                'patterns': pm.datasets.ToyDatasetEyeLink().custom_read_kwargs['gaze']['patterns'],
                'schema': pm.datasets.ToyDatasetEyeLink().custom_read_kwargs['gaze']['schema'],
            },
            pl.DataFrame(
                data={
                    'time': [
                        2154556, 2154557, 2154560, 2154564, 2154596, 2154598, 2154599, 2154695,
                        2154696, 2339227, 2339245, 2339246, 2339271, 2339272, 2339290, 2339291,
                    ],
                    'pupil': [
                        778.0, 778.0, 777.0, 778.0, 784.0, 784.0, 784.0, 798.0,
                        799.0, 619.0, 621.0, 622.0, 617.0, 617.0, 618.0, 618.0,
                    ],
                    'pixel': [
                        [138.1, 132.8], [138.2, 132.7], [137.9, 131.6], [138.1, 131.0],
                        [139.6, 132.1], [139.5, 131.9], [139.5, 131.8], [147.2, 134.4],
                        [147.3, 134.1], [673.2, 523.8], [629.0, 531.4], [629.9, 531.9],
                        [639.4, 531.9], [639.0, 531.9], [637.6, 531.4], [637.3, 531.2],
                    ],
                    'trial_id': [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, None],
                    'point_id': 3 * [None] + [0, 1, 2, 3] + [None, 0] + [0, 0, 1, 2] + [0, 1, None],
                    'screen_id': [None, 0, 1] + 13 * [None],
                    'task': [None] + 2 * ['reading'] + 12 * ['judo'] + [None],
                },
                schema={
                    'time': pl.Int64,
                    'pupil': pl.Float64,
                    'task': pl.Utf8,
                    'trial_id': pl.Int64,
                    'point_id': pl.Int64,
                    'screen_id': pl.Int64,
                    'pixel': pl.List(pl.Float64),
                },
            ),
            id='eyelink_asc_mono_pattern_list',
        ),
        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_2khz_example.asc',
                'patterns': 'eyelink',
            },
            pl.from_dict(
                data={
                    'time': [
                        2154556.5, 2154557.0, 2154560.5, 2154564.0, 2154596.0, 2154598.5, 2154599.0,
                        2154695.0, 2154696.0, 2339227.0, 2339245.0, 2339246.0, 2339271.5, 2339272.0,
                        2339290.0, 2339291.0,
                    ],
                    'pupil': [
                        778.0, 778.0, 777.0, 778.0, 784.0, 784.0, 784.0, 798.0,
                        799.0, 619.0, 621.0, 622.0, 617.0, 617.0, 618.0, 618.0,
                    ],
                    'pixel': [
                        [138.1, 132.8], [138.2, 132.7], [137.9, 131.6], [138.1, 131.0],
                        [139.6, 132.1], [139.5, 131.9], [139.5, 131.8], [147.2, 134.4],
                        [147.3, 134.1], [673.2, 523.8], [629.0, 531.4], [629.9, 531.9],
                        [639.4, 531.9], [639.0, 531.9], [637.6, 531.4], [637.3, 531.2],
                    ],
                },
                schema={
                    'time': pl.Float64,
                    'pupil': pl.Float64,
                    'pixel': pl.List(pl.Float64),
                },
            ),
            id='eyelink_asc_mono_2khz_pattern_eyelink',
        ),
        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_no_dummy_example.asc',
                'patterns': 'eyelink',
            },
            pl.from_dict(
                data={
                    'time': [
                        643197, 643199, 643201, 643203, 643205, 643207, 643209, 647791, 647793, 647795, 647797, 647799, 647801, 647803, 647805, 647807, 647809, 647811, 647813, 647815, 647817, 647819, 647821, 647823, 647825, 647827, 647829, 647831, 647833, 647835, 647837, 647839, 647841, 647843, 647845, 647847, 647849, 647851, 647853, 647855, 647857, 647859, 647861, 647863, 647865, 647867, 647869, 647871, 647873, 647875, 647877, 647879, 647881, 647883, 647885, 647887, 647889, 647891, 647893, 647895, 647897, 647899, 647901, 647903, 647905, 647907, 647909, 647911, 647913, 647915, 647917, 647919, 647921, 647923, 647925, 647927, 647929, 647931, 647933, 647935, 647937, 647939, 647941, 647943, 647945, 647947, 647949, 647951, 647953, 647955, 647957, 647959, 647961, 647963, 647965, 647967, 647969, 647971, 647973, 647975, 647977, 647979, 647981, 647983, 647985, 647987, 647989, 647991, 647993, 647995, 647997, 647999, 648001, 648003, 648005, 648007, 648009, 648011, 648013, 648015, 648017, 648019, 648021, 648023, 648025, 648027, 648029, 648031, 648033, 648035, 648037, 648039, 648041, 648043, 648045, 648047, 648049, 648051, 648053, 648055, 648057, 648059, 648061, 648063, 648065, 648067, 648069, 648071, 648073, 648075, 648077, 648079, 648081, 648083, 648085, 648087, 648089, 648091, 648093, 648095, 648097, 648099, 648101, 648103, 648105, 648107, 648109, 648111, 648113, 648115, 648117, 648119, 648121, 648123, 648125, 648127, 648129, 648131, 648133, 648135, 648137, 648139, 648141, 648143, 648145, 648147, 648149, 648151, 648153, 648155, 648157, 648159, 648161, 648163, 648165, 648167, 648169, 648171, 648173, 648175, 648177, 648179, 648181, 648183, 648185, 648187, 648189, 648191, 648193, 648195, 648197, 648199, 648201, 648203, 648205, 648207, 648209, 648211, 648213, 648215, 648217, 648219, 648221, 648223, 648225, 648227, 648229, 648231, 648233, 648235, 648237, 648239, 648241, 648243, 648245, 648247, 648249, 648251, 651171, 651173, 651175, 651177, 651179, 651181, 651183, 651185, 651187, 651189, 651191, 651193, 651195, 651197, 651199, 651201, 651203, 651205, 651207, 651209, 651211, 651213, 651215, 651217, 651219, 651221, 651223, 651225, 651227, 651229, 651231, 651233, 651235, 651237, 651239, 651241, 651243, 651245, 651247, 651249, 651251, 651253, 651255, 651257, 651259, 651261, 651263, 651265, 651267, 651269, 651271, 651273, 651275, 651277, 651279, 651281, 651283, 651285, 651287,
                    ],
                    'pupil': [
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 802.0, 801.0, 800.0, 825.0, 852.0, 862.0, 869.0, 878.0, 890.0, 905.0, 912.0, 910.0, 911.0, 922.0, 926.0, 929.0, 939.0, 942.0, 946.0, 953.0, 960.0, 963.0, 965.0, 967.0, 965.0, 966.0, 971.0, 973.0, 980.0, 982.0, 983.0, 984.0, 988.0, 991.0, 992.0, 999.0, 1003.0, 1007.0, 1011.0, 1011.0, 1009.0, 1006.0, 1008.0, 1012.0, 1014.0, 1013.0, 1015.0, 1016.0, 1019.0, 1021.0, 1026.0, 1028.0, 1033.0, 1046.0, 1060.0, 1072.0, 1085.0, 1095.0, 1108.0, 1114.0, 1112.0, 1112.0, 1115.0, 1118.0, 1123.0, 1125.0, 1127.0, 1128.0, 1131.0, 1131.0, 1129.0, 1132.0, 1135.0, 1138.0, 1141.0, 1140.0, 1140.0, 1143.0, 1146.0, 1149.0, 1156.0, 1153.0, 1153.0, 1154.0, 1159.0, 1168.0, 1162.0, 1161.0, 1163.0, 1165.0, 1167.0, 1170.0, 1172.0, 1172.0, 1172.0, 1175.0, 1178.0, 1181.0, 1178.0, 1181.0, 1188.0, 1188.0, 1188.0, 1188.0, 1185.0, 1183.0, 1183.0, 1183.0, 1187.0, 1191.0, 1190.0, 1189.0, 1190.0, 1191.0, 1191.0, 1190.0, 1190.0, 1193.0, 1193.0, 1192.0, 1193.0, 1194.0, 1196.0, 1200.0, 1205.0, 1207.0, 1206.0, 1202.0, 1199.0, 1201.0, 1205.0, 1201.0, 1203.0, 1209.0, 1211.0, 1211.0, 1209.0, 1207.0, 1207.0, 1207.0, 1207.0, 1207.0, 1210.0, 1216.0, 1218.0, 1215.0, 1212.0, 1215.0, 1217.0, 1215.0, 1214.0, 1216.0, 1218.0, 1218.0, 1214.0, 1216.0, 1216.0, 1214.0, 1211.0, 1216.0, 1216.0, 1214.0, 1210.0, 1211.0, 1214.0, 1212.0, 1213.0, 1214.0, 1217.0, 1218.0, 1212.0, 1211.0, 1212.0, 1212.0, 1213.0, 1212.0, 1210.0, 1210.0, 1211.0, 1215.0, 1217.0, 1214.0, 1214.0, 1213.0, 1212.0, 1209.0, 1212.0, 1212.0, 1214.0, 1214.0, 1213.0, 1212.0, 1208.0, 1201.0, 1204.0, 1209.0, 1208.0, 1208.0, 1209.0, 1209.0, 1209.0, 1206.0, 1206.0, 1208.0, 1210.0, 1209.0, 1210.0, 1209.0, 1207.0, 1208.0, 1207.0, 1206.0, 1206.0, 1207.0, 1208.0, 1210.0, 1211.0, 1212.0, 1212.0, 1211.0, 1208.0, 1207.0, 1208.0, 1213.0, 1215.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 331.0, 362.0, 445.0,
                    ],
                    'pixel': [
                        [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [839.1, 583.5], [838.6, 582.1], [838.3, 578.8], [845.4, 589.3], [851.2, 607.0], [852.1, 616.2], [853.4, 626.9], [856.2, 634.7], [858.8, 640.5], [860.2, 648.6], [862.3, 653.5], [864.3, 654.8], [864.6, 662.3], [863.3, 668.1], [864.7, 670.3], [865.8, 673.6], [867.1, 679.9], [867.4, 683.8], [867.3, 688.0], [869.2, 691.8], [870.4, 692.5], [871.0, 694.1], [870.7, 698.8], [872.1, 700.7], [872.4, 701.4], [870.4, 702.4], [870.0, 706.5], [871.4, 708.0], [872.0, 708.5], [871.0, 713.3], [871.3, 716.1], [872.3, 717.1], [872.6, 718.9], [872.7, 719.5], [872.4, 719.0], [871.6, 719.6], [870.6, 720.8], [870.7, 721.8], [871.1, 725.2], [870.5, 724.6], [869.4, 723.2], [869.0, 722.4], [869.8, 724.1], [870.6, 726.4], [870.0, 726.2], [869.5, 725.9], [869.9, 726.0], [871.2, 726.4], [872.5, 725.9], [872.3, 725.9], [870.3, 727.1], [866.4, 726.7], [857.9, 725.8], [845.9, 725.6], [831.7, 723.9], [810.8, 717.9], [790.7, 710.0], [771.9, 702.5], [757.0, 694.1], [746.0, 688.5], [738.7, 681.7], [732.5, 674.9], [728.9, 664.9], [726.0, 654.5], [724.6, 645.6], [725.3, 637.5], [727.8, 633.8], [730.3, 633.3], [731.2, 633.5], [732.5, 632.9], [733.3, 633.9], [733.9, 636.5], [734.4, 638.1], [736.8, 639.5], [737.4, 640.6], [736.8, 641.6], [736.8, 641.1], [736.7, 640.4], [735.7, 638.5], [735.0, 636.4], [735.2, 636.7], [733.9, 635.0], [732.7, 632.7], [732.4, 631.7], [732.5, 631.5], [732.5, 631.0], [731.8, 628.3], [730.2, 626.8], [729.9, 626.4], [729.1, 625.2], [728.1, 624.2], [727.7, 622.7], [727.4, 620.6], [725.4, 618.2], [724.8, 617.8], [724.9, 617.6], [724.0, 616.4], [723.4, 615.8], [723.0, 614.9], [723.2, 613.6], [723.4, 611.6], [723.5, 610.9], [723.0, 610.0], [722.5, 608.6], [721.5, 606.7], [720.8, 604.4], [721.5, 602.9], [721.9, 601.9], [721.6, 602.0], [721.8, 601.4], [721.6, 601.1], [721.3, 599.4], [721.1, 599.1], [719.9, 598.1], [718.4, 596.3], [717.2, 594.2], [716.5, 593.0], [716.0, 593.0], [715.3, 593.3], [714.4, 592.5], [712.8, 592.3], [712.2, 592.1], [712.9, 591.0], [713.4, 588.3], [713.4, 587.4], [713.3, 586.6], [713.0, 586.0], [713.1, 584.1], [713.2, 581.6], [711.2, 580.7], [710.6, 580.6], [710.5, 580.8], [709.9, 580.4], [708.8, 579.5], [708.7, 578.2], [708.6, 577.0], [708.4, 576.5], [708.4, 576.8], [707.5, 576.9], [706.3, 575.8], [705.9, 574.6], [704.9, 573.0], [704.3, 572.1], [704.6, 573.1], [704.4, 573.1], [703.7, 571.7], [703.8, 570.7], [703.9, 570.0], [703.6, 569.7], [702.6, 568.7], [701.6, 567.3], [702.4, 567.1], [702.3, 566.8], [701.7, 565.4], [701.9, 564.2], [701.4, 564.3], [700.2, 563.7], [699.9, 562.9], [699.8, 561.4], [699.9, 560.1], [699.9, 560.3], [699.6, 560.5], [699.3, 560.0], [698.8, 559.2], [698.3, 556.9], [698.7, 556.9], [699.2, 558.4], [698.8, 558.3], [698.7, 557.9], [698.8, 556.6], [698.6, 555.0], [698.0, 554.8], [697.2, 554.8], [696.0, 554.8], [695.8, 555.0], [696.8, 555.1], [696.3, 554.9], [695.6, 554.4], [695.5, 554.1], [695.3, 554.0], [695.3, 553.4], [695.6, 552.7], [695.9, 552.3], [695.5, 551.7], [695.1, 551.3], [694.9, 550.7], [694.3, 551.2], [693.2, 551.2], [693.1, 550.8], [693.1, 549.9], [692.7, 549.0], [692.2, 548.4], [690.9, 547.8], [690.9, 546.2], [691.6, 546.0], [691.7, 546.0], [691.6, 546.1], [690.9, 546.2], [690.7, 545.2], [691.2, 544.9], [691.0, 545.5], [690.3, 545.7], [688.9, 545.1], [688.3, 544.9], [688.9, 545.0], [688.6, 545.0], [688.3, 545.1], [687.9, 545.4], [687.7, 545.3], [687.3, 545.0], [687.1, 544.4], [686.9, 543.6], [687.0, 543.5], [687.1, 543.7], [686.0, 542.8], [685.3, 542.6], [685.5, 543.4], [686.2, 542.8], [686.3, 542.7], [685.6, 543.7], [685.6, 543.4], [685.8, 542.9], [685.7, 543.6], [685.1, 543.6], [683.9, 543.1], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [127.5, 621.5], [125.9, 602.8], [120.3, 551.3],
                    ],
                },
                schema={
                    'time': pl.Int64,
                    'pupil': pl.Float64,
                    'pixel': pl.List(pl.Float64),
                },
            ),
            id='eyelink_asc_mono_no_dummy_pattern_eyelink',
        ),
    ],
)
def test_from_asc_has_shape_and_schema(kwargs, expected_frame):
    gaze = pm.gaze.from_asc(**kwargs)

    print(gaze.frame['time'].to_list())
    print(gaze.frame['pupil'].to_list())
    print(gaze.frame['pixel'].to_list())
    assert_frame_equal(gaze.frame, expected_frame, check_column_order=False)


@pytest.mark.parametrize(
    ('kwargs', 'exception', 'message'),
    [
        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_example.asc',
                'patterns': 'foobar',
            },
            ValueError,
            "unknown pattern key 'foobar'. Supported keys are: eyelink",
            id='unknown_pattern',
        ),
    ],
)
def test_from_asc_raises_exception(kwargs, exception, message):
    with pytest.raises(exception) as excinfo:
        pm.gaze.from_asc(**kwargs)

    msg, = excinfo.value.args
    assert msg == message


@pytest.mark.parametrize(
    ('file', 'metadata'),
    [
        pytest.param(
            'tests/files/eyelink_monocular_example.asc',
            {
                'width_px': 1280,
                'height_px': 1024,
                'sampling_rate': 1000.0,
                'left': True,
                'right': False,
                'model': 'EyeLink Portable Duo',
                'version': '6.12',
                'vendor': 'EyeLink',
                'mount': 'Desktop',
            },
            id='1khz',
        ),
        pytest.param(
            'tests/files/eyelink_monocular_2khz_example.asc',
            {
                'width_px': 1280,
                'height_px': 1024,
                'sampling_rate': 2000.0,
                'left': True,
                'right': False,
                'model': 'EyeLink Portable Duo',
                'version': '6.12',
                'vendor': 'EyeLink',
                'mount': 'Desktop',
            },
            id='2khz',
        ),
        pytest.param(
            'tests/files/eyelink_monocular_no_dummy_example.asc',
            {
                'width_px': 1920,
                'height_px': 1080,
                'sampling_rate': 500.0,
                'left': True,
                'right': False,
                'model': 'EyeLink 1000 Plus',
                'version': '5.50',
                'vendor': 'EyeLink',
                'mount': 'Desktop',
            },
            id='500hz_no_dummy',
        ),
    ],
)
def test_from_asc_fills_in_experiment_metadata(file, metadata):
    gaze = pm.gaze.from_asc(file, experiment=None)
    assert gaze.experiment.screen.width_px == metadata['width_px']
    assert gaze.experiment.screen.height_px == metadata['height_px']
    assert gaze.experiment.eyetracker.sampling_rate == metadata['sampling_rate']
    assert gaze.experiment.eyetracker.left is metadata['left']
    assert gaze.experiment.eyetracker.right is metadata['right']
    assert gaze.experiment.eyetracker.model == metadata['model']
    assert gaze.experiment.eyetracker.version == metadata['version']
    assert gaze.experiment.eyetracker.vendor == metadata['vendor']
    assert gaze.experiment.eyetracker.mount == metadata['mount']


@pytest.mark.parametrize(
    ('experiment_kwargs', 'issues'),
    [
        pytest.param(
            {
                'screen_width_px': 1920,
                'screen_height_px': 1080,
                'sampling_rate': 1000,
            },
            ['Screen resolution: (1920, 1080) vs. (1280, 1024)'],
            id='screen_resolution',
        ),
        pytest.param(
            {
                'eyetracker': pm.EyeTracker(sampling_rate=500),
            },
            ['Sampling rate: 500 vs. 1000.0'],
            id='eyetracker_sampling_rate',
        ),
        pytest.param(
            {
                'eyetracker': pm.EyeTracker(
                    left=False,
                    right=True,
                    sampling_rate=1000,
                    mount='Desktop',
                ),
            },
            [
                'Left eye tracked: False vs. True',
                'Right eye tracked: True vs. False',
            ],
            id='eyetracker_tracked_eye',
        ),
        pytest.param(
            {
                'eyetracker': pm.EyeTracker(
                    vendor='Tobii',
                    model='Tobii Pro Spectrum',
                    version='1.0',
                    sampling_rate=1000,
                    left=True,
                    right=False,
                ),
            },
            [
                'Eye tracker vendor: Tobii vs. EyeLink',
                'Eye tracker model: Tobii Pro Spectrum vs. EyeLink Portable Duo',
                'Eye tracker software version: 1.0 vs. 6.12',
            ],
            id='eyetracker_vendor_model_version',
        ),
        pytest.param(
            {
                'eyetracker': pm.EyeTracker(
                    mount='Remote',
                    sampling_rate=1000,
                    vendor='EyeLink',
                    model='EyeLink Portable Duo',
                    version='6.12',
                ),
            },
            ['Mount configuration: Remote vs. Desktop'],
            id='eyetracker_mount',
        ),
    ],
)
def test_from_asc_detects_mismatches_in_experiment_metadata(experiment_kwargs, issues):
    with pytest.raises(ValueError) as excinfo:
        pm.gaze.from_asc(
            'tests/files/eyelink_monocular_example.asc',
            experiment=pm.Experiment(**experiment_kwargs),
        )

    msg, = excinfo.value.args
    expected_msg = 'Experiment metadata does not match the metadata in the ASC file:\n'
    expected_msg += '\n'.join(f'- {issue}' for issue in issues)
    assert msg == expected_msg
