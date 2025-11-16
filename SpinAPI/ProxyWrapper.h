/////////////////////////////////////////////////////////////////////////
// ProxyWrapper class (SpinAPI Module)
// ------------------
// Proxy class that will wrap around the armadillo matrix types. whilst allowing for other infomation to be stored in the return object 
// In principle this is shoul result in very little changes to the tasks.
//
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2019 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////

#include <memory>
#include <armadillo>

class ProxyWrapper
{
private:
    std::unique_ptr<arma::sp_cx_mat> implSparse;
    std::unique_ptr<arma::cx_mat> implDense;
    int impl;
public:
    ProxyWrapper();
    //set implementation
    template<typename t, typename... Args>
    void Set(Args&& ... args) 
    {
        if (t==arma::sp_cx_mat)
        {
            implSparse = std::make_unique<t>(std::forward<Args>(args) ... );
            impl = 0;
        }
        else
        {
            implDense = std::make_unique<t>(std::forward<Args>(args) ... );
            impl = 1;
        }
    }
};