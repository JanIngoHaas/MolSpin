/////////////////////////////////////////////////////////////////////////
// TaskStaticSSPowderSpectraNakajimaZwanzig implementation (RunSection module)  developed by Irina Anisimova.
//
// Notes: no pulses implemented at the current state
//
// Molecular Spin Dynamics Software - developed by Luca Gerhards.
// (c) 2022 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <iomanip>
#include "TaskStaticSSPowderSpectraNakajimaZwanzig.h"
#include "Transition.h"
#include "Settings.h"
#include "State.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "Spin.h"
#include "Interaction.h"
#include "ObjectParser.h"
#include "Operator.h"
#include "Pulse.h"

namespace RunSection
{
	// -----------------------------------------------------
	// TaskStaticSSPowderSpectraNakajimaZwanzig Constructors and Destructor
	// -----------------------------------------------------
	TaskStaticSSPowderSpectraNakajimaZwanzig::TaskStaticSSPowderSpectraNakajimaZwanzig(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection), timestep(1.0), totaltime(1.0e+4), reactionOperators(SpinAPI::ReactionOperatorType::Haberkorn)
	{
	}

	TaskStaticSSPowderSpectraNakajimaZwanzig::~TaskStaticSSPowderSpectraNakajimaZwanzig()
	{
	}

	bool TaskStaticSSPowderSpectraNakajimaZwanzig::BuildNakajimaZwanzigLiouvillian(auto &_i, SpinAPI::SpinSpace &_space, const arma::cx_mat &_H, arma::cx_mat &_A, arma::cx_mat &_eigenvec)
	{
		_space.UseSuperoperatorSpace(false);

		arma::vec eigen_val;
		arma::eig_sym(eigen_val, _eigenvec, _H);
		arma::cx_mat eig_val_mat = arma::diagmat(arma::conv_to<arma::cx_mat>::from(eigen_val));

		arma::cx_mat one;
		one.set_size(arma::size(_H));
		one.eye();
		arma::cx_mat lambda = (arma::kron(eig_val_mat, one) - arma::kron(one, eig_val_mat.st()));

		arma::cx_mat R;
		R.set_size(arma::size(lambda));
		R.zeros();

		auto accumulate_terms = [&](const std::vector<arma::cx_mat> &tensors, const std::vector<double> &input_ampl, const std::vector<double> &input_tau, int terms, int def_g) -> bool {
			if (tensors.empty())
				return true;

			std::vector<double> ampl_list = input_ampl;
			std::vector<double> tau_c_list = input_tau;
			if (tau_c_list.empty())
				return true;
			if (ampl_list.empty())
				ampl_list.push_back(0.0);

			auto amp_at = [&](int idx) -> double {
				if (ampl_list.empty())
					return 0.0;
				if (idx < 0)
					return ampl_list.front();
				if (idx < static_cast<int>(ampl_list.size()))
					return ampl_list[static_cast<size_t>(idx)];
				return ampl_list.front();
			};

			if (def_g == 0)
			{
				if (ampl_list.size() < tau_c_list.size())
					ampl_list.resize(tau_c_list.size(), ampl_list.front());
				if (tau_c_list.size() < ampl_list.size())
					tau_c_list.resize(ampl_list.size(), tau_c_list.front());
			}

			arma::cx_mat SpecDens;
			SpecDens.set_size(arma::size(lambda));
			SpecDens.zeros();
			arma::cx_mat tmp_R;
			tmp_R.set_size(arma::size(lambda));
			tmp_R.zeros();

			const int num_op = static_cast<int>(tensors.size());
			if (terms == 1)
			{
				if (def_g == 1)
				{
					const std::complex<double> tau = static_cast<std::complex<double>>(tau_c_list.front());
					for (int k = 0; k < num_op; k++)
					{
						const std::complex<double> ampl = static_cast<std::complex<double>>(amp_at(k) * amp_at(k));
						SpecDens.zeros();
						if (!this->ConstructSpecDensSpecificSpectra(ampl, tau, lambda, SpecDens))
							return false;
						tmp_R.zeros();
						if (!this->NakajimaZwanzigtensorSpectra(tensors[static_cast<size_t>(k)], tensors[static_cast<size_t>(k)], SpecDens, tmp_R))
							return false;
						R += tmp_R;
					}
				}
				else
				{
					SpecDens.zeros();
					if (!this->ConstructSpecDensGeneralSpectra(ampl_list, tau_c_list, lambda, SpecDens))
						return false;
					for (int k = 0; k < num_op; k++)
					{
						tmp_R.zeros();
						if (!this->NakajimaZwanzigtensorSpectra(tensors[static_cast<size_t>(k)], tensors[static_cast<size_t>(k)], SpecDens, tmp_R))
							return false;
						R += tmp_R;
					}
				}
			}
			else
			{
				if (def_g == 1)
				{
					const std::complex<double> tau = static_cast<std::complex<double>>(tau_c_list.front());
					for (int k = 0; k < num_op; k++)
					{
						for (int s = 0; s < num_op; s++)
						{
							const std::complex<double> ampl = static_cast<std::complex<double>>(amp_at(k) * amp_at(s));
							SpecDens.zeros();
							if (!this->ConstructSpecDensSpecificSpectra(ampl, tau, lambda, SpecDens))
								return false;
							tmp_R.zeros();
							if (!this->NakajimaZwanzigtensorSpectra(tensors[static_cast<size_t>(k)], tensors[static_cast<size_t>(s)], SpecDens, tmp_R))
								return false;
							R += tmp_R;
						}
					}
				}
				else
				{
					SpecDens.zeros();
					if (!this->ConstructSpecDensGeneralSpectra(ampl_list, tau_c_list, lambda, SpecDens))
						return false;
					for (int k = 0; k < num_op; k++)
					{
						for (int s = 0; s < num_op; s++)
						{
							tmp_R.zeros();
							if (!this->NakajimaZwanzigtensorSpectra(tensors[static_cast<size_t>(k)], tensors[static_cast<size_t>(s)], SpecDens, tmp_R))
								return false;
							R += tmp_R;
						}
					}
				}
			}
			return true;
		};

		this->Log() << "Starting NZ relaxation matrix construction for internal powder orientation." << std::endl;

		for (auto interaction = (*_i)->interactions_cbegin(); interaction < (*_i)->interactions_cend(); interaction++)
		{
			bool interaction_relaxation = false;
			const bool has_relax_flag = (*interaction)->Properties()->Get("relaxation", interaction_relaxation);
			if (has_relax_flag && !interaction_relaxation)
				continue;

			int terms = 1;
			int def_g = 1;
			int def_multexpo = 0;
			int ops = 1;
			int coeff = 0;
			(void)(*interaction)->Properties()->Get("terms", terms);
			(void)(*interaction)->Properties()->Get("def_g", def_g);
			(void)(*interaction)->Properties()->Get("def_multexpo", def_multexpo);
			(void)(*interaction)->Properties()->Get("ops", ops);
			(void)(*interaction)->Properties()->Get("coeff", coeff);

			if (def_multexpo == 1)
			{
				this->Log() << "NZ def_multexpo=1 is not supported in powder-NZ task yet for interaction \"" << (*interaction)->Name() << "\"; skipping this interaction." << std::endl;
				continue;
			}

			std::vector<double> tau_c_list;
			std::vector<double> ampl_list;
			if (!(*interaction)->Properties()->GetList("tau_c", tau_c_list))
			{
				this->Log() << "NZ interaction \"" << (*interaction)->Name() << "\" has no tau_c list; skipping." << std::endl;
				continue;
			}
			if (!(*interaction)->Properties()->GetList("g", ampl_list))
			{
				this->Log() << "NZ interaction \"" << (*interaction)->Name() << "\" has no g list; skipping." << std::endl;
				continue;
			}

			if ((*interaction)->Type() == SpinAPI::InteractionType::SingleSpin)
			{
				auto group1 = (*interaction)->Group1();
				for (auto s1 = group1.cbegin(); s1 < group1.cend(); s1++)
				{
					std::vector<arma::cx_mat> tensors;
					if (ops == 1)
					{
						arma::cx_mat Sx, Sy, Sz;
						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sx()), *s1, Sx))
							return false;
						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sy()), *s1, Sy))
							return false;
						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sz()), *s1, Sz))
							return false;
						tensors.push_back(Sx);
						tensors.push_back(Sy);
						tensors.push_back(Sz);
					}
					else
					{
						arma::vec static_field = (*interaction)->Field();
						arma::cx_vec complex_field = arma::conv_to<arma::cx_vec>::from(static_field);

						arma::cx_mat T0_rank_0, T0_rank_2, Tm1, Tp1, Tm2, Tp2;
						if (!_space.LRk0TensorT0((*s1), complex_field, T0_rank_0))
							return false;
						if (!_space.LRk2SphericalTensorT0((*s1), complex_field, T0_rank_2))
							return false;
						if (!_space.LRk2SphericalTensorTm1((*s1), complex_field, Tm1))
							return false;
						if (!_space.LRk2SphericalTensorTp1((*s1), complex_field, Tp1))
							return false;
						if (!_space.LRk2SphericalTensorTm2((*s1), complex_field, Tm2))
							return false;
						if (!_space.LRk2SphericalTensorTp2((*s1), complex_field, Tp2))
							return false;

						tensors.push_back(T0_rank_0);
						tensors.push_back(T0_rank_2);
						tensors.push_back(-Tm1);
						tensors.push_back(-Tp1);
						tensors.push_back(Tm2);
						tensors.push_back(Tp2);
					}

					for (auto &tensor : tensors)
						tensor = (_eigenvec.t() * tensor * _eigenvec);

					if (!accumulate_terms(tensors, ampl_list, tau_c_list, terms, def_g))
						return false;
				}
			}
			else if ((*interaction)->Type() == SpinAPI::InteractionType::DoubleSpin)
			{
				auto group1 = (*interaction)->Group1();
				auto group2 = (*interaction)->Group2();
				for (auto s1 = group1.cbegin(); s1 < group1.cend(); s1++)
				{
					for (auto s2 = group2.cbegin(); s2 < group2.cend(); s2++)
					{
						std::vector<arma::cx_mat> tensors;
						if (ops == 1)
						{
							arma::cx_mat Sx1, Sy1, Sz1, Sx2, Sy2, Sz2;
							if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sx()), *s1, Sx1))
								return false;
							if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sy()), *s1, Sy1))
								return false;
							if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sz()), *s1, Sz1))
								return false;
							if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s2)->Sx()), *s2, Sx2))
								return false;
							if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s2)->Sy()), *s2, Sy2))
								return false;
							if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s2)->Sz()), *s2, Sz2))
								return false;

							tensors.push_back(Sx1 * Sx2);
							tensors.push_back(Sx1 * Sy2);
							tensors.push_back(Sx1 * Sz2);
							tensors.push_back(Sy1 * Sx2);
							tensors.push_back(Sy1 * Sy2);
							tensors.push_back(Sy1 * Sz2);
							tensors.push_back(Sz1 * Sx2);
							tensors.push_back(Sz1 * Sy2);
							tensors.push_back(Sz1 * Sz2);
						}
						else
						{
							arma::cx_mat T0_rank_0, T0_rank_2, Tm1, Tp1, Tm2, Tp2;
							if (!_space.BlRk0TensorT0(*s1, *s2, T0_rank_0))
								return false;
							if (!_space.BlRk2SphericalTensorT0(*s1, *s2, T0_rank_2))
								return false;
							if (!_space.BlRk2SphericalTensorTm1(*s1, *s2, Tm1))
								return false;
							if (!_space.BlRk2SphericalTensorTp1(*s1, *s2, Tp1))
								return false;
							if (!_space.BlRk2SphericalTensorTm2(*s1, *s2, Tm2))
								return false;
							if (!_space.BlRk2SphericalTensorTp2(*s1, *s2, Tp2))
								return false;

							tensors.push_back(T0_rank_0);
							tensors.push_back(T0_rank_2);
							tensors.push_back(-Tm1);
							tensors.push_back(-Tp1);
							tensors.push_back(Tm2);
							tensors.push_back(Tp2);

							SpinAPI::Tensor inTensor(0);
							if ((*interaction)->Properties()->Get("tensor", inTensor) && coeff == 1)
							{
								auto ATensor = (*interaction)->CouplingTensor();
								arma::cx_mat A(3, 3, arma::fill::zeros);
								arma::cx_vec Am(9, arma::fill::zeros);
								A = arma::conv_to<arma::cx_mat>::from((ATensor->LabFrame()));
								Am(0) = (1.0 / sqrt(3.0)) * (A(0, 0) + A(1, 1) + A(2, 2));
								Am(1) = (1.0 / sqrt(6.0)) * (3.0 * A(2, 2) - (A(0, 0) + A(1, 1) + A(2, 2)));
								Am(2) = 0.5 * (A(0, 2) + A(2, 0) + ((arma::cx_double(0.0, 1.0)) * (A(1, 2) + A(2, 1))));
								Am(3) = -0.5 * (A(0, 2) + A(2, 0) - ((arma::cx_double(0.0, 1.0)) * (A(1, 2) + A(2, 1))));
								Am(4) = 0.5 * (A(0, 0) - A(1, 1) - ((arma::cx_double(0.0, 1.0)) * (A(0, 1) + A(1, 0))));
								Am(5) = 0.5 * (A(0, 0) - A(1, 1) + ((arma::cx_double(0.0, 1.0)) * (A(0, 1) + A(1, 0))));

								tensors[0] = Am(0) * tensors[0];
								tensors[1] = Am(1) * tensors[1];
								tensors[2] = Am(3) * tensors[2];
								tensors[3] = Am(2) * tensors[3];
								tensors[4] = Am(4) * tensors[4];
								tensors[5] = Am(5) * tensors[5];
							}
						}

						for (auto &tensor : tensors)
							tensor = (_eigenvec.t() * tensor * _eigenvec);

						if (!accumulate_terms(tensors, ampl_list, tau_c_list, terms, def_g))
							return false;
					}
				}
			}
		}

		arma::cx_mat lhs;
		arma::cx_mat rhs;
		arma::cx_mat H_SS;
		_space.SuperoperatorFromLeftOperator(eig_val_mat, lhs);
		_space.SuperoperatorFromRightOperator(eig_val_mat, rhs);
		H_SS = lhs - rhs;
		_A = arma::cx_double(0.0, -1.0) * H_SS;

		arma::cx_mat K;
		arma::cx_mat Klhs;
		arma::cx_mat Krhs;
		if (!_space.TotalReactionOperator(K))
		{
			this->Log() << "Warning: Failed to obtain matrix representation of the reaction operators!" << std::endl;
		}
		K = (_eigenvec.t() * K * _eigenvec);
		_space.SuperoperatorFromLeftOperator(K, Klhs);
		_space.SuperoperatorFromRightOperator(K, Krhs);
		_A -= (Klhs + Krhs);

		arma::cx_mat O_SS;
		for (auto t = (*_i)->operators_cbegin(); t != (*_i)->operators_cend(); t++)
		{
			_space.UseSuperoperatorSpace(true);
			if (_space.RelaxationOperatorFrameChange((*t), _eigenvec, O_SS))
			{
				_A += O_SS;
			}
			else
			{
				this->Log() << "There is a problem with operator \"" << (*t)->Name() << "\". Please check.\n";
			}
			_space.UseSuperoperatorSpace(false);
		}

		_A += R;
		_space.UseSuperoperatorSpace(true);
		return true;
	}

	bool TaskStaticSSPowderSpectraNakajimaZwanzig::ConvertSuperspaceToLab(auto &_space, const arma::cx_vec &_rho_vec_eig, const arma::cx_mat &_eigenvec, arma::cx_vec &_rho_vec_lab)
	{
		arma::cx_mat rho_eig;
		if (!_space.OperatorFromSuperspace(_rho_vec_eig, rho_eig))
			return false;
		arma::cx_mat rho_lab = _eigenvec * rho_eig * _eigenvec.t();
		return _space.OperatorToSuperspace(rho_lab, _rho_vec_lab);
	}

	// -----------------------------------------------------
	// TaskStaticSSPowderSpectraNakajimaZwanzig protected methods
	// -----------------------------------------------------
	bool TaskStaticSSPowderSpectraNakajimaZwanzig::RunLocal()
	{
		this->Log() << "Running method StaticSS-PowderSpectra-NakajimaZwanzig." << std::endl;

		// If this is the first step, write first part of header to the data file
		if (this->RunSettings()->CurrentStep() == 1)
		{
			this->WriteHeader(this->Data());
		}

		// Decline density matrix and density vector variables
		arma::cx_mat rho0;
		arma::cx_vec rho0vec;

		// Obtain spin systems
		auto systems = this->SpinSystems();

		// Loop through all SpinSystems
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{
			// Make sure we have an initial state
			auto initial_states = (*i)->InitialState();
			if (initial_states.size() < 1)
			{
				this->Log() << "Skipping SpinSystem \"" << (*i)->Name() << "\" as no initial state was specified." << std::endl;
				continue;
			}

			this->Log() << "\nStarting with SpinSystem \"" << (*i)->Name() << "\"." << std::endl;

			// Obtain a SpinSpace to describe the system
			SpinAPI::SpinSpace space(*(*i));
			space.UseSuperoperatorSpace(true);
			space.SetReactionOperatorType(this->reactionOperators);

			std::vector<double> weights;
			weights = (*i)->Weights();

			// Normalize the weights
			double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
			if (sum_weights > 0)
			{
				for (double &weight : weights)
				{
					weight /= sum_weights;
				}
			}

			// Get the initial state
			if (weights.size() > 1)
			{
				this->Log() << "Using weighted density matrix for initial state. Be sure that the sum of weights equals to 1." << std::endl;
				// Get the initial state
				int counter = 0;
				for (auto j = initial_states.cbegin(); j != initial_states.cend(); j++)
				{
					arma::cx_mat tmp_rho0;
					if ((*j) == nullptr)
					{
						this->Log() << "Failed to obtain weighted thermal initial state for SpinSystem \"" << (*i)->Name() << "\". Weighted thermal initial states are not supported." << std::endl;
						continue;
					}
					if (!space.GetState(*j, tmp_rho0))
					{
						this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\", initial state of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
						continue;
					}

					if (j == initial_states.cbegin())
					{
						this->Log() << "State: \"" << (*j)->Name() << "\", Weight:\"" << weights[0] << "\"." << std::endl;
						rho0 = weights[0] * tmp_rho0;
						counter += 1;
					}
					else
					{
						this->Log() << "State: \"" << (*j)->Name() << "\", Weight:\"" << weights[counter] << "\"." << std::endl;
						rho0 += weights[counter] * tmp_rho0;
						counter += 1;
					}
				}
			}
			else
			// Get the initial state without weights
			{
				for (auto j = initial_states.cbegin(); j != initial_states.cend(); j++)
				{
					arma::cx_mat tmp_rho0;

					// Get the initial state in thermal equilibrium
					if ((*j) == nullptr) // "Thermal initial state"
					{
						this->Log() << "Initial state = thermal " << std::endl;

						// Get the thermalhamiltonianlist
						std::vector<std::string> thermalhamiltonian_list = (*i)->ThermalHamiltonianList();

						this->Log() << "ThermalHamiltonianList = [";
						for (size_t j = 0; j < thermalhamiltonian_list.size(); j++)
						{
							this->Log() << thermalhamiltonian_list[j];
							if (j < thermalhamiltonian_list.size() - 1)
								this->Log() << ", "; // Add a comma between elements
						}
						this->Log() << "]" << std::endl;

						// Get temperature
						double temperature = (*i)->Temperature();
						this->Log() << "Temperature = " << temperature << "K" << std::endl;

						// Get the initial state with thermal equilibrium
						if (!space.GetThermalState(space, temperature, thermalhamiltonian_list, tmp_rho0))
						{
							this->Log() << "Failed to obtain projection matrix onto thermal state, initial state of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
							continue;
						}
					}
					else // Get the initial state without thermal equilibrium
					{
						if (!space.GetState(*j, tmp_rho0))
						{
							this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\", initial state of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
							continue;
						}
					}

					// Obtain the initial density matrix
					if (j == initial_states.cbegin())
					{
						rho0 = tmp_rho0;
					}
					else
					{
						rho0 += tmp_rho0;
					}
				}
			}

			rho0 /= arma::trace(rho0); // The density operator should have a trace of 1

			// Convert initial state to superoperator space
			if (!space.OperatorToSuperspace(rho0, rho0vec))
			{
				this->Log() << "Failed to convert initial state density operator to superspace." << std::endl;
				continue;
			}

			// Read in input parameters

			// Read the method from the input file
			std::string Method;
			if (!this->Properties()->Get("method", Method))
			{
				this->Log() << "Failed to obtain an input for a Method. Please specify method = timeinf or method = timeevo." << std::endl;
			}

			// Read if the result should be integrated or not if method.
			bool integration = false;
			if (!this->Properties()->Get("integration", integration))
			{
				this->Log() << "Failed to obtain an input for an integtation. Plese use integration = true/false. Using integration = false by default. " << std::endl;
			}
			this->Log() << "Integration of the yield in time on a grid  = " << integration << std::endl;

			// Read CIDSP from the input file
			bool CIDSP = false;
			if (!this->Properties()->Get("cidsp", CIDSP))
			{
				this->Log() << "Failed to obtain an input for a CIDSP. Plese use cidsp = true/false. Using cidsp = false by default. " << std::endl;
			}

			// Read in the number of points on the sampling grid
			int numPoints = 1000;
			if (!this->Properties()->Get("powdersamplingpoints", numPoints))
			{
				this->Log() << "Failed to obtain an input for a number of sampling points. Plese use powdersamplingpoints = N. Using powdersamplingpoints = 1000 by default. " << std::endl;
			}

			double Printedtime = 0;

			// Construct grid
			std::vector<std::tuple<double, double, double>> grid;
			if (!this->CreateUniformGrid(numPoints, grid))
			{
				this->Log() << "Failed to obtain an Uniform grid." << std::endl;
			}

			// Method Propagation to infinity
			if (Method.compare("timeinf") == 0)
			{
				// Perform the calculation
				this->Log() << "Ready to perform calculation." << std::endl;

				this->Log() << "Method = " << Method << std::endl;

				std::vector<arma::cx_vec> rho_tmp(numPoints);
				for (auto &v : rho_tmp)
					v.zeros(size(rho0vec));

				arma::cx_vec integral;
				integral.zeros(size(rho0vec));

				// Initialize a first step
				arma::cx_vec rhovec = rho0vec;

				for (int grid_num = 0; grid_num < numPoints; ++grid_num)
				{
					auto [theta, phi, weight] = grid[grid_num];

					// Make the option without powdering from inside possible
					if (numPoints <= 1)
					{
						theta = 0.0;
						phi = 0.0;
						weight = 1.0;
					}

					// Construct the rotation matrix
					arma::mat Rot_mat;
					double gamma = 0;
					if (!this->CreateRotationMatrix(gamma, theta, phi, Rot_mat))
					{
						this->Log() << "Failed to obtain an Lebedev grid." << std::endl;
					}

					// Construct the hamiltonian H0
					std::vector<std::string> HamiltonianH0list;
					if (!this->Properties()->GetList("hamiltonianh0list", HamiltonianH0list, ','))
					{
						this->Log() << "Failed to obtain an input for a HamiltonianH0." << std::endl;
					}

						space.UseSuperoperatorSpace(false);
						arma::sp_cx_mat H0;
						if (!space.BaseHamiltonianRotated_SA(HamiltonianH0list, Rot_mat, H0))
						{
							this->Log() << "Failed to obtain Hamiltonian in superspace." << std::endl;
							continue;
						}

						std::vector<std::string> HamiltonianH1list;
						if (!this->Properties()->GetList("hamiltonianh1list", HamiltonianH1list, ','))
						{
							this->Log() << "Failed to obtain an input for a HamiltonianH1." << std::endl;
						}

						arma::sp_cx_mat H1;
						if (!space.ThermalHamiltonian(HamiltonianH1list, H1))
						{
							this->Log() << "Failed to obtain Hamiltonian in superspace." << std::endl;
							continue;
						}

						arma::cx_mat H = arma::conv_to<arma::cx_mat>::from(H0 + H1);
						arma::cx_mat A_nz;
						arma::cx_mat eigen_vec;
						if (!this->BuildNakajimaZwanzigLiouvillian(i, space, H, A_nz, eigen_vec))
						{
							this->Log() << "Failed to build NZ Liouvillian for current powder orientation." << std::endl;
							continue;
						}

						space.UseSuperoperatorSpace(true);

						arma::cx_mat rho0_eig = (eigen_vec.t() * rho0 * eigen_vec);
						rho0_eig /= arma::trace(rho0_eig);
						arma::cx_vec rho0vec_eig;
						if (!space.OperatorToSuperspace(rho0_eig, rho0vec_eig))
						{
							this->Log() << "Failed to convert initial state to superspace in NZ eigenbasis." << std::endl;
							continue;
						}

						arma::cx_vec result_eig = -solve(A_nz, rho0vec_eig);
						arma::cx_vec result_lab;
						result_lab.zeros(size(rho0vec));
						if (!this->ConvertSuperspaceToLab(space, result_eig, eigen_vec, result_lab))
						{
							this->Log() << "Failed to transform NZ result back to lab frame." << std::endl;
							continue;
						}

						// Integrate over all grid points in the lab basis.
						integral += weight * result_lab;
					}

				rhovec = integral;

				if (!this->ProjectAndPrintOutputLineInf(i, space, rhovec, Printedtime, this->timestep, CIDSP, this->Data(), this->Log()))
					this->Log() << "Could not project the state vector and print the result into a file" << std::endl;

				this->Log() << "Done with calculation." << std::endl;
			}
			// Method TIME EVOLUTION
			else if (Method.compare("timeevo") == 0)
			{

				if (!this->totaltime == 0)
				{
					// Perform the calculation
					this->Log() << "Ready to perform calculation." << std::endl;

					this->Log() << "Method = " << Method << std::endl;

					// Create a holder vector for an averaged density
					int firststep = 0;
					unsigned int time_steps = static_cast<unsigned int>(std::abs(this->totaltime / this->timestep));
					std::vector<arma::cx_vec> rho_avg(time_steps + 1);
					for (auto &v : rho_avg)
						v.zeros(size(rho0vec));

					for (int grid_num = 0; grid_num < numPoints; ++grid_num)
					{
						auto [theta, phi, weight] = grid[grid_num];

						// Make the option without powdering from inside possible
						if (numPoints <= 1)
						{
							theta = 0.0;
							phi = 0.0;
							weight = 1.0;
						}

							// Create rotation matrix
							arma::mat Rot_mat;
							double gamma = 0;
							if (!this->CreateRotationMatrix(gamma, theta, phi, Rot_mat))
							{
							this->Log() << "Failed to obtain an Lebedev grid." << std::endl;
						}

						// Create Hamiltonian H0
						std::vector<std::string> HamiltonianH0list;
						if (!this->Properties()->GetList("hamiltonianh0list", HamiltonianH0list, ','))
						{
							this->Log() << "Failed to obtain an input for a HamiltonianH0." << std::endl;
						}

							space.UseSuperoperatorSpace(false);
							arma::sp_cx_mat H0;
							if (!space.BaseHamiltonianRotated_SA(HamiltonianH0list, Rot_mat, H0))
							{
								this->Log() << "Failed to obtain Hamiltonian in superspace." << std::endl;
								continue;
							}

							std::vector<std::string> HamiltonianH1list;
							if (!this->Properties()->GetList("hamiltonianh1list", HamiltonianH1list, ','))
							{
								this->Log() << "Failed to obtain an input for a HamiltonianH1." << std::endl;
							}

							arma::sp_cx_mat H1;
							if (!space.ThermalHamiltonian(HamiltonianH1list, H1))
							{
								this->Log() << "Failed to obtain Hamiltonian in superspace." << std::endl;
								continue;
							}

							arma::cx_mat H = arma::conv_to<arma::cx_mat>::from(H0 + H1);
							arma::cx_mat A_nz;
							arma::cx_mat eigen_vec;
							if (!this->BuildNakajimaZwanzigLiouvillian(i, space, H, A_nz, eigen_vec))
							{
								this->Log() << "Failed to build NZ Liouvillian for current powder orientation." << std::endl;
								continue;
							}

							space.UseSuperoperatorSpace(true);

							arma::cx_mat rho0_eig = (eigen_vec.t() * rho0 * eigen_vec);
							rho0_eig /= arma::trace(rho0_eig);
							arma::cx_vec rho0vec_eig;
							if (!space.OperatorToSuperspace(rho0_eig, rho0vec_eig))
							{
								this->Log() << "Failed to convert initial state to superspace in NZ eigenbasis." << std::endl;
								continue;
							}

							arma::cx_vec rhoavg_n_eig;
							rhoavg_n_eig.zeros(size(rho0vec_eig));

							std::pair<arma::sp_cx_mat, arma::cx_vec> G;
							arma::sp_cx_mat A_sp = arma::conv_to<arma::sp_cx_mat>::from(arma::expmat(A_nz * this->timestep));
							G = std::pair<arma::sp_cx_mat, arma::cx_vec>(A_sp, rho0vec_eig);

							for (unsigned int n = firststep; n <= time_steps; n++)
							{
								arma::cx_vec out_vec_eig;
								if (n == 0)
								{
									out_vec_eig = G.second;
								}
								else
								{
									arma::cx_vec rhovec_eig = G.first * G.second;
									if (integration)
									{
										rhoavg_n_eig += this->timestep * (G.second + rhovec_eig) / 2;
									}
									G.second = rhovec_eig;
									if (!rhoavg_n_eig.is_zero(0))
									{
										out_vec_eig = rhoavg_n_eig;
									}
									else
									{
										out_vec_eig = rhovec_eig;
									}
								}

								arma::cx_vec out_vec_lab;
								out_vec_lab.zeros(size(rho0vec));
								if (!this->ConvertSuperspaceToLab(space, out_vec_eig, eigen_vec, out_vec_lab))
								{
									this->Log() << "Failed to transform NZ time-domain result back to lab frame." << std::endl;
									continue;
								}

								rho_avg[n] += weight * out_vec_lab;
							}
						}

					for (unsigned int n = firststep; n <= time_steps; n++)
					{
						if (!this->ProjectAndPrintOutputLine(i, space, rho_avg[n], Printedtime, this->timestep, n, CIDSP, this->Data(), this->Log()))
							this->Log() << "Could not project the state vector and print the result into a file" << std::endl;
					}

				}

				this->Log() << "Done with calculation." << std::endl;
			}
			else
			{
				this->Log() << "Undefined spectroscopy method. Please choose between timeinf or timeevo methods." << std::endl;
			}

			this->Log() << "\nDone with SpinSystem \"" << (*i)->Name() << "\"" << std::endl;
		}

		// Terminate the line in the data file after iteration through all spin systems
		this->Data() << std::endl;

		return true;
	}

	// Writes the header of the data file (but can also be passed to other streams)
	void TaskStaticSSPowderSpectraNakajimaZwanzig::WriteHeader(std::ostream &_stream)
	{
		_stream << "Step ";
		_stream << "Time ";
		this->WriteStandardOutputHeader(_stream);

		std::vector<std::string> spinList;
		bool CIDSP = false;
		int m;

		// Get header for each spin system
		auto systems = this->SpinSystems();
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{

			if (this->Properties()->GetList("spinlist", spinList, ','))
			{
				for (auto l = (*i)->spins_cbegin(); l != (*i)->spins_cend(); l++)
				{
					std::string spintype;

					(*l)->Properties()->Get("type", spintype);

					for (m = 0; m < (int)spinList.size(); m++)
					{

						if ((*l)->Name() == spinList[m])
						{
							// Yields are written per transition
							// bool CIDSP = false;
							if (this->Properties()->Get("cidsp", CIDSP) && CIDSP == true)
							{
								// Write each transition name
								auto transitions = (*i)->Transitions();
								for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
								{
									_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".yield"
											<< ".Ix ";
									_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".yield"
											<< ".Iy ";
									_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".yield"
											<< ".Iz ";
									_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".yield"
											<< ".Ip ";
									_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".yield"
											<< ".Im ";
								}
							}
							else
							{
								// Write each state name
								auto states = (*i)->States();
								_stream << (*i)->Name() << "." << (*l)->Name() << ".Ix ";
								_stream << (*i)->Name() << "." << (*l)->Name() << ".Iy ";
								_stream << (*i)->Name() << "." << (*l)->Name() << ".Iz ";
								_stream << (*i)->Name() << "." << (*l)->Name() << ".Ip ";
								_stream << (*i)->Name() << "." << (*l)->Name() << ".Im ";
							}
						}
					}
				}
			}
		}
		_stream << std::endl;
	}

	// Validation
	bool TaskStaticSSPowderSpectraNakajimaZwanzig::Validate()
	{

		double inputTimestep = 0.0;
		double inputTotaltime = 0.0;

		// Get timestep
		if (this->Properties()->Get("timestep", inputTimestep))
		{
			if (std::isfinite(inputTimestep) && inputTimestep > 0.0)
			{
				this->timestep = inputTimestep;
			}
			else
			{
				this->Log() << "# WARNING: undefined timestep, using by default 0.1 ns!" << std::endl;
				this->timestep = 0.1;
			}
		}

		// Get totaltime
		if (this->Properties()->Get("totaltime", inputTotaltime))
		{
			if (std::isfinite(inputTotaltime) && inputTotaltime >= 0.0)
			{
				this->totaltime = inputTotaltime;
			}
			else
			{
				this->Log() << "# ERROR: invalid total time!" << std::endl;
				return false;
			}
		}

		// Get the reacton operator type
		std::string str;
		if (this->Properties()->Get("reactionoperators", str))
		{
			if (str.compare("haberkorn") == 0)
			{
				this->reactionOperators = SpinAPI::ReactionOperatorType::Haberkorn;
				this->Log() << "Setting reaction operator type to Haberkorn." << std::endl;
			}
			else if (str.compare("lindblad") == 0)
			{
				this->reactionOperators = SpinAPI::ReactionOperatorType::Lindblad;
				this->Log() << "Setting reaction operator type to Lindblad." << std::endl;
			}
			else
			{
				this->Log() << "Warning: Unknown reaction operator type specified. Using default reaction operators." << std::endl;
			}
		}

		return true;
	}

	arma::cx_vec TaskStaticSSPowderSpectraNakajimaZwanzig::ComputeRhoDot(double t, arma::sp_cx_mat &L, arma::cx_vec &K, arma::cx_vec RhoNaught)
	{
		arma::cx_vec ReturnVec(L.n_rows);
		RhoNaught = RhoNaught + K;
		ReturnVec = L * RhoNaught;
		return ReturnVec;
	}

	bool TaskStaticSSPowderSpectraNakajimaZwanzig::ProjectAndPrintOutputLine(auto &_i, SpinAPI::SpinSpace &_space, arma::cx_vec &_rhovec, double &_printedtime, double _timestep, unsigned int &_n, bool &_cidsp, std::ostream &_datastream, std::ostream &_logstream)
	{
		arma::cx_mat rho0;

		// Convert the resulting density operator back to its Hilbert space representation
		if ((!_space.OperatorFromSuperspace(_rhovec, rho0)) && (_n == 0))
		{
			_logstream << "Failed to convert resulting superspace-vector back to native Hilbert space." << std::endl;
			return false;
		}

		// Get nuclei of interest for CIDNP spectrum
		arma::cx_mat Iprojx;
		arma::cx_mat Iprojy;
		arma::cx_mat Iprojz;
		arma::cx_mat Iprojp;
		arma::cx_mat Iprojm;

		std::vector<std::string> spinList;

		if (_n == 0)
			_logstream << "CIDSP = " << _cidsp << std::endl;

		// Save the current step
		_datastream << this->RunSettings()->CurrentStep() << " ";
		// Save the current time
		_datastream << std::setprecision(12) << _printedtime + (_n * _timestep) << " ";
		this->WriteStandardOutput(_datastream);

		if (this->Properties()->GetList("spinlist", spinList, ','))
		{

			for (auto l = (*_i)->spins_cbegin(); l != (*_i)->spins_cend(); l++)
			{
				for (int m = 0; m < (int)spinList.size(); m++)
				{
					if ((*l)->Name() == spinList[m])
					{
						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sx()), (*l), Iprojx))
						{
							return false;
						}

						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sy()), (*l), Iprojy))
						{
							return false;
						}

						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sz()), (*l), Iprojz))
						{
							return false;
						}

						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sp()), (*l), Iprojp))
						{
							return false;
						}

						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sm()), (*l), Iprojm))
						{
							return false;
						}

						arma::cx_mat P;

						// There are two result modes - either write results per transition  if CIDSP is true or for each defined state if CIDSP is false

						if (_cidsp == true)
						{
							// Loop through all defind transitions
							auto transitions = (*_i)->Transitions();
							for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
							{
								// Make sure that there is a state object
								if ((*j)->SourceState() == nullptr)
									continue;

								if ((!_space.GetState((*j)->SourceState(), P)) && (_n == 0))
								{
									_logstream << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\" of SpinSystem \"" << (*_i)->Name() << "\"." << std::endl;
									continue;
								}

								// Return the yield for this transition
								_datastream << std::real(arma::trace(Iprojx * (*j)->Rate() * P * rho0)) << " ";
								_datastream << std::real(arma::trace(Iprojy * (*j)->Rate() * P * rho0)) << " ";
								_datastream << std::real(arma::trace(Iprojz * (*j)->Rate() * P * rho0)) << " ";
								_datastream << std::real(arma::trace(Iprojp * (*j)->Rate() * P * rho0)) << " ";
								_datastream << std::real(arma::trace(Iprojm * (*j)->Rate() * P * rho0)) << " ";
							}
						}
						else if (_cidsp == false)
						{
							// Return the yield for this state - note that no reaction rates are included here.
							_datastream << std::real(arma::trace(Iprojx * rho0)) << " ";
							_datastream << std::real(arma::trace(Iprojy * rho0)) << " ";
							_datastream << std::real(arma::trace(Iprojz * rho0)) << " ";
							_datastream << std::real(arma::trace(Iprojp * rho0)) << " ";
							_datastream << std::real(arma::trace(Iprojm * rho0)) << " ";
						}
					}
				}
			}
		}
		else
		{
			if (_n == 0)
				_logstream << "No nucleus was specified for projection" << std::endl;
			return false;
		}

		_datastream << std::endl;

		return true;
	}

	bool TaskStaticSSPowderSpectraNakajimaZwanzig::ProjectAndPrintOutputLineInf(auto &_i, SpinAPI::SpinSpace &_space, arma::cx_vec &_rhovec, double &_printedtime, double _timestep, bool &_cidsp, std::ostream &_datastream, std::ostream &_logstream)
	{
		arma::cx_mat rho0;

		// Convert the resulting density operator back to its Hilbert space representation
		if ((!_space.OperatorFromSuperspace(_rhovec, rho0)))
		{
			_logstream << "Failed to convert resulting superspace-vector back to native Hilbert space." << std::endl;
			return false;
		}

		// Get nuclei of interest for CIDNP spectrum
		arma::cx_mat Iprojx;
		arma::cx_mat Iprojy;
		arma::cx_mat Iprojz;
		arma::cx_mat Iprojp;
		arma::cx_mat Iprojm;

		std::vector<std::string> spinList;

		_logstream << "CIDSP = " << _cidsp << std::endl;

		// Save the current step
		_datastream << this->RunSettings()->CurrentStep() << " ";
		// Save the current time
		_datastream << "inf" << " ";
		this->WriteStandardOutput(_datastream);

		if (this->Properties()->GetList("spinlist", spinList, ','))
		{

			for (auto l = (*_i)->spins_cbegin(); l != (*_i)->spins_cend(); l++)
			{
				for (int m = 0; m < (int)spinList.size(); m++)
				{
					if ((*l)->Name() == spinList[m])
					{
						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sx()), (*l), Iprojx))
						{
							return false;
						}

						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sy()), (*l), Iprojy))
						{
							return false;
						}

						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sz()), (*l), Iprojz))
						{
							return false;
						}

						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sp()), (*l), Iprojp))
						{
							return false;
						}

						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sm()), (*l), Iprojm))
						{
							return false;
						}

						arma::cx_mat P;

						// There are two result modes - either write results per transition  if CIDSP is true or for each defined state if CIDSP is false

						if (_cidsp == true)
						{
							// Loop through all defind transitions
							auto transitions = (*_i)->Transitions();
							for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
							{
								// Make sure that there is a state object
								if ((*j)->SourceState() == nullptr)
									continue;

								if ((!_space.GetState((*j)->SourceState(), P)))
								{
									_logstream << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\" of SpinSystem \"" << (*_i)->Name() << "\"." << std::endl;
									continue;
								}

								// Return the yield for this transition
								_datastream << std::real(arma::trace(Iprojx * (*j)->Rate() * P * rho0)) << " ";
								_datastream << std::real(arma::trace(Iprojy * (*j)->Rate() * P * rho0)) << " ";
								_datastream << std::real(arma::trace(Iprojz * (*j)->Rate() * P * rho0)) << " ";
								_datastream << std::real(arma::trace(Iprojp * (*j)->Rate() * P * rho0)) << " ";
								_datastream << std::real(arma::trace(Iprojm * (*j)->Rate() * P * rho0)) << " ";
							}
						}
						else if (_cidsp == false)
						{
							// Return the yield for this state - note that no reaction rates are included here.
							_datastream << std::real(arma::trace(Iprojx * rho0)) << " ";
							_datastream << std::real(arma::trace(Iprojy * rho0)) << " ";
							_datastream << std::real(arma::trace(Iprojz * rho0)) << " ";
							_datastream << std::real(arma::trace(Iprojp * rho0)) << " ";
							_datastream << std::real(arma::trace(Iprojm * rho0)) << " ";
						}
					}
				}
			}
		}
		else
		{
			_logstream << "No nucleus was specified for projection" << std::endl;
			return false;
		}

		return true;
	}

	bool TaskStaticSSPowderSpectraNakajimaZwanzig::CreateRotationMatrix(double &_alpha, double &_beta, double &_gamma, arma::mat &_R) const
	{
		arma::mat R1 = {
			{std::cos(_alpha), -std::sin(_alpha), 0.0},
			{std::sin(_alpha), std::cos(_alpha), 0.0},
			{0.0, 0.0, 1.0}};

		arma::mat R2 = {
			{std::cos(_beta), 0.0, std::sin(_beta)},
			{0.0, 1.0, 0.0},
			{-std::sin(_beta), 0.0, std::cos(_beta)}};

		arma::mat R3 = {
			{std::cos(_gamma), -std::sin(_gamma), 0.0},
			{std::sin(_gamma), std::cos(_gamma), 0.0},
			{0.0, 0.0, 1.0}};

		_R = R1 * R2 * R3;

		return true;
	}

	bool TaskStaticSSPowderSpectraNakajimaZwanzig::CreateUniformGrid(int &_Npoints, std::vector<std::tuple<double, double, double>> &_uniformGrid) const
	{
		std::vector<double> theta(_Npoints);
		std::vector<double> phi(_Npoints);
		std::vector<double> weight(_Npoints);

		_uniformGrid.resize(_Npoints);

		const double golden = M_PI * (1.0 + std::sqrt(5.0)); // not standart golden angle

		for (int i = 0; i < _Npoints; ++i)
		{
			double index = static_cast<double>(i) + 0.5;

			theta[i] = std::acos(1.0 - index / _Npoints);		  // hemisphere
			phi[i] = golden * index;							  // hemisphere
			weight[i] = std::sin(theta[i]) * 2 * M_PI / _Npoints; // 2 * pi for hemisphere
			_uniformGrid[i] = {theta[i], phi[i], weight[i]};
		}

		return true;
	}

	bool TaskStaticSSPowderSpectraNakajimaZwanzig::CreateCustomGrid(int &_Npoints, std::vector<std::tuple<double, double, double>> &_Grid) const
	{
		std::vector<double> theta(_Npoints * _Npoints);
		std::vector<double> phi(_Npoints * _Npoints);
		std::vector<double> weight(_Npoints * _Npoints);

		_Grid.resize(_Npoints * _Npoints);

		int idx = 0;
		for (int k = 0; k < _Npoints; ++k)
		{
			double u = (k + 0.5) / _Npoints; // cosine-spaced
			double th = acos(u);			 // θ

			for (int j = 0; j < _Npoints; ++j)
			{
				double ph = (j + 0.5) * (M_PI / 2.0) / _Npoints; // uniform φ

				theta[idx] = th;
				phi[idx] = ph;

				weight[idx] = (M_PI / 2.0 / _Npoints) * (1.0 / _Npoints); // Δφ * Δ(cosθ)
				_Grid[idx] = {theta[idx], phi[idx], weight[idx]};
				idx++;
			}
		}

		return true;
	}

	bool TaskStaticSSPowderSpectraNakajimaZwanzig::NakajimaZwanzigtensorSpectra(const arma::cx_mat &_op1, const arma::cx_mat &_op2, const arma::cx_mat &_specdens, arma::cx_mat &_NakajimaZwanzigtensor)
	{
		_NakajimaZwanzigtensor *= 0.0;

		arma::cx_mat _op1_SS = _specdens;
		arma::cx_mat _op2_SS = _specdens;
		arma::cx_mat one = arma::eye<arma::cx_mat>(arma::size(_op1));

		_op1_SS = arma::kron((_op1).t(), one) - arma::kron(one, (_op1.t()).st());
		_op2_SS = arma::kron((_op2), one) - arma::kron(one, (_op2).st());

		_NakajimaZwanzigtensor = arma::cx_double(-1.00, 0.00) * _op1_SS * _specdens.t() * _op2_SS;
		return true;
	}

	bool TaskStaticSSPowderSpectraNakajimaZwanzig::ConstructSpecDensGeneralSpectra(const std::vector<double> &_ampl_list, const std::vector<double> &_tau_c_list, const arma::cx_mat &_omega, arma::cx_mat &_specdens)
	{
		arma::cx_vec spectral_entries = _omega.diag();
		spectral_entries.zeros();

		for (auto ii = 0; ii < (int)_omega.n_cols; ii++)
		{
			for (auto jj = 0; jj < (int)_tau_c_list.size(); jj++)
			{
				spectral_entries(ii) += (static_cast<std::complex<double>>(_ampl_list[jj])) / ((1.00 / (static_cast<std::complex<double>>(_tau_c_list[jj]))) + arma::cx_double(0.0, -1.00) * _omega(ii, ii));
			}
		}

		_specdens = arma::diagmat(arma::conv_to<arma::cx_mat>::from(spectral_entries));
		return true;
	}

	bool TaskStaticSSPowderSpectraNakajimaZwanzig::ConstructSpecDensSpecificSpectra(const std::complex<double> &_ampl, const std::complex<double> &_tau_c, const arma::cx_mat &_omega, arma::cx_mat &_specdens)
	{
		arma::cx_vec spectral_entries = _omega.diag();
		for (auto ii = 0; ii < (int)_omega.n_cols; ii++)
		{
			spectral_entries(ii) = _ampl / ((1.00 / _tau_c) + arma::cx_double(0.0, -1.00) * _omega(ii, ii));
		}

		_specdens = arma::diagmat(arma::conv_to<arma::cx_mat>::from(spectral_entries));
		return true;
	}

	// -----------------------------------------------------
}
