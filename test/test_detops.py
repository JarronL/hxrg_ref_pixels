from ref_pixels import detops

def test_multiaccum_frames():

    ma = detops.multiaccum()
    patterns = list(ma._pattern_settings.keys())

    for patt in patterns:
        ma.read_mode = patt

        # Calculate number of total frames in a group
        nf_group = ma.nf + ma.nd1 + ma.nd2 + ma.nd3
        if ma.read_mode=='RAPID':
            assert nf_group==1
        if "BRIGHT" in ma.read_mode:
            assert nf_group==2
        if "SHALLOW" in ma.read_mode:
            assert nf_group==5
        if "MEDIUM" in ma.read_mode:
            assert nf_group==10
        if "DEEP" in ma.read_mode:
            assert nf_group==20
