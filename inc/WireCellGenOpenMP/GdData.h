#ifndef _OMP_GDDATA_H_
#define _OMP_GDDATA_H_

namespace WireCell
{  
  namespace GenOpenMP
  {
    struct GdData 
    {
      double p_ct ;
      double t_ct ;
	    double p_sigma ;
	    double t_sigma ;
	    double charge ;
    };

    struct DBin 
    {
      double minval ;
      double binsize ;
      int  nbins ;
    };
  } //namespace GenOpenMP
} //namespace WireCell

#endif
