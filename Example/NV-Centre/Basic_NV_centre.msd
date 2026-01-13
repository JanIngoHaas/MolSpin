SpinSystem GroundState
{
    Spin e1
    {
        spin = 1/2;
        type = electron;
        tensor = isotropic(2);
    }

    Spin e2 
    {
        spin = 1/2;
        type = electron; 
        tensor = isotropic(2);
    }
    
    //nitrogen-14
    Spin N14
    { 
        spin = 1;
        type = nucleus;
        tensor = isotropic;
    }

    Interaction ZFS
    {
        type = zfs;
    }
}