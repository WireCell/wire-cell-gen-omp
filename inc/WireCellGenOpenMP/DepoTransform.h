/** Make a frame from depos using an ImpactTransform.
 */

#ifndef WIRECELL_GENOPENMP_DEPOTRANSFORM
#define WIRECELL_GENOPENMP_DEPOTRANSFORM

#include "WireCellIface/IDepoFramer.h"
#include "WireCellIface/IConfigurable.h"
#include "WireCellIface/IRandom.h"
#include "WireCellIface/IPlaneImpactResponse.h"
#include "WireCellIface/IAnodePlane.h"
#include "WireCellIface/WirePlaneId.h"
#include "WireCellIface/IDepo.h"
#include "WireCellUtil/Logging.h"

namespace WireCell {
    namespace GenOpenMP {

        class DepoTransform : public IDepoFramer, public IConfigurable {
           public:
            DepoTransform();
            virtual ~DepoTransform();

            virtual bool operator()(const input_pointer& in, output_pointer& out);

            virtual void configure(const WireCell::Configuration& cfg);
            virtual WireCell::Configuration default_configuration() const;

            /// dummy depo modifier
            /// used for the application of the charge scaling bases on dQdx calibration
            /// see the detailed implementation in larwirecell or uboonecode
            virtual IDepo::pointer modify_depo(WirePlaneId wpid, IDepo::pointer depo) { return depo; }

           private:
            IAnodePlane::pointer m_anode;
            IRandom::pointer m_rng;
            std::vector<IPlaneImpactResponse::pointer> m_pirs;

            double m_start_time;
            double m_readout_time;
            double m_tick;
            double m_drift_speed;
            double m_nsigma;
            int m_frame_count;
            std::string m_transform;
            Log::logptr_t l;
        };
    }  // namespace GenOpenMP
}  // namespace WireCell

#endif
