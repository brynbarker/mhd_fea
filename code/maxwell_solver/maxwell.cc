#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_nedelec_sz.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/constrained_linear_operator.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <fstream>
#include <cmath>

// solves dh/dt - curl (u X h) + nu curl ( curl (h) ) + grad q+ beta (div h)e_1 = f
// div h = 0

using namespace dealii;

template <int dim>
class Maxwell
{
    public:
        Maxwell(const unsigned int degree, 
                const unsigned int n_global_refinements,  
                const Tensor<1, dim> velocity_field);


        void run();

    private:
        void setup_grid();
        void setup_dofs();
        void assemble_system();
        void output_results() const;

        const unsigned int degree;
        unsigned int n_global_refinements;

        double time_step;
        double end_time;
        double current_time;
        int    time_step_number;

        double nu;
        double beta;

        Tensor<1, dim> vel_field;

        Triangulation<dim> triangulation;
        FESystem<dim>      fe;
        DoFHandler<dim>    dof_handler;

        AffineConstraints<double> constraints;

        BlockSparsityPattern      sparsity_pattern;
        BlockSparsityPattern      unconstrained_sparsity_pattern;
        BlockSparseMatrix<double> system_matrix;
        //BlockSparseMatrix<double> mass_matrix;

        BlockSparseMatrix<double> unconstrained_system_matrix;
        BlockSparseMatrix<double> unconstrained_mass_matrix;

        BlockSparsityPattern      preconditioner_sparsity_pattern;
        BlockSparseMatrix<double> preconditioner_matrix;
        SparseMatrix<double>      mass_p_matrix;

        BlockVector<double> current_solution;
        BlockVector<double> previous_solution;
        BlockVector<double> system_rhs;
        BlockVector<double> load_vector;
};

template<int dim>
class ExactSolution : public Function<dim>
{
    public:
        ExactSolution() 
            : Function<dim>(dim+1) // 3
        {}

        virtual void 
        vector_value(const Point<dim> &p, 
                     Vector<double> &  values) const override
        {
            const double time = this->get_time();
            int order  = 3;

            if (order == 0)
            {
                values(0) = 1.0;
                values(1) = 1.0;
            }
            if (order == 1)
            {
                values(0) = -p(1);
                values(1) = p(0);
            }
            if (order == 2)
            {
                values(0) = -p(1)*p(1);
                values(1) = p(0)*p(0);
            }
            if (order == 3)
            {
                values(0) = time*cos(p(1));
                values(1) = time*sin(p(0));
            }
            values(2) = 0;
        }
};

template <int dim>
class ForcingFunction : public Function<dim>
{
    public:
        ForcingFunction(double nu, Tensor<1, dim> &v_field)
            : Function<dim>(dim+1) 
            , nu(nu)
            , v_field(v_field)// 3
        {}

        virtual void 
        vector_value(const Point<dim> &p, 
                     Vector<double> &  values) const override
        {
            const double time = this->get_time();
            int order  = 3;

            if (order == 0)
            {
                values(0) = 0.0;
                values(1) = 0.0;
            }
            if (order == 1)
            {
                values(0) = -v_field[1];
                values(1) = v_field[0];
            }
            if (order == 2)
            {
                values(0) = 2*(nu-v_field[1]*p(1));
                values(1) = -2*(nu-v_field[0]*p(0));
            }
            if (order == 3)
            {
                values(0) = cos(p(1))*(nu*time+1)-time*v_field[1]*sin(p(1));
                values(1) = sin(p(0))*(nu*time+1)+time*v_field[0]*cos(p(0));
            }
            values(2) = 0;
        }
    private:
        double nu;
        Tensor<1,dim> &v_field;
};

template<class MatrixType, class PreconditionerType>
class InverseMatrix : public Subscriptor
{
    public:
        InverseMatrix(const MatrixType & m,
                      const PreconditionerType &preconditioner)
            : matrix(&m)
            , preconditioner(&preconditioner)
        {}

        void vmult(Vector<double> &sol, const Vector<double> &vec) const
        {
            SolverControl solver_control(vec.size(), 1e-6*vec.l2_norm());
            SolverGMRES<Vector<double>> gmres(solver_control);
            sol = 0;
            gmres.solve(*matrix, sol, vec, *preconditioner);
        }

    private:
        const SmartPointer<const MatrixType> matrix;
        const SmartPointer<const PreconditionerType> preconditioner;
};

template<class InverseType>
class SchurComplement : public Subscriptor
{
    public:
        SchurComplement(
                const BlockSparseMatrix<double> &system_matrix,
                const InverseType &A_inv)
            : system_matrix(&system_matrix)
            , A_inv(&A_inv)
            , tmp1(system_matrix.block(0,0).m())
            , tmp2(system_matrix.block(0,0).m())
        {}

        void vmult(Vector<double> &sol, const Vector<double> &vec) const
        {
            system_matrix->block(0,1).vmult(tmp1, vec);
            A_inv->vmult(tmp2, tmp1);
            system_matrix->block(1,0).vmult(sol, tmp2);
        }

    private:
        const SmartPointer<const BlockSparseMatrix<double> > system_matrix;
        const SmartPointer<const InverseType> A_inv;

        mutable Vector<double> tmp1, tmp2;
};

template <int dim>
Maxwell<dim>::Maxwell(const unsigned int degree,
                      const unsigned int n_global_refinements,
                      const Tensor<1, dim> velocity_field)
    : degree(degree)
    , n_global_refinements(n_global_refinements)
    , time_step(std::pow(0.1, n_global_refinements))
    , current_time(0.0)
    , time_step_number(1)
    , nu(1.0)
    , beta(1.0)
    , vel_field(velocity_field)
    , fe(FE_NedelecSZ<dim>(degree+1), 1, FE_Q<dim>(degree), 1)
    , dof_handler(triangulation)
{}


template <int dim>
void
Maxwell<dim>::setup_grid()
{
  GridGenerator::hyper_cube(triangulation);

  triangulation.refine_global(n_global_refinements);
}


template <int dim>
void Maxwell<dim>::setup_dofs()
{
    // make sure system is clear
    system_matrix.clear();
    unconstrained_mass_matrix.clear();
    unconstrained_system_matrix.clear();
    preconditioner_matrix.clear();
    mass_p_matrix.clear();

    dof_handler.distribute_dofs(fe);

    // group together the velocity components 
    // separate from the lagrange multiplier
    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;

    DoFRenumbering::component_wise(dof_handler);

    std::vector<types::global_dof_index> dofs_per_component(
            dim+1, types::global_dof_index(0));

    DoFTools::count_dofs_per_component(dof_handler,
                                       dofs_per_component, 
                                       true); // this means there will be no dublicates

    const unsigned int n_h = dofs_per_component[0];
    const unsigned int n_q = dofs_per_component[dim];

    // it is overkill to use exact solution here
    ExactSolution<dim> exact_solution;
    exact_solution.set_time(current_time);

    {
      constraints.clear();
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      
      // FE_Nedelec boundary condition.
      VectorTools::project_boundary_values_curl_conforming_l2(
          dof_handler,
          0,
          exact_solution,
          0,
          constraints,
          StaticMappingQ1<dim>::mapping);
  
      FEValuesExtractors::Scalar q(dim);
      // Lagrange multiplier boundary conditions
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               exact_solution,
                                               constraints,
                                               fe.component_mask(q));
    }


    constraints.close();

    // setup sparsity patterns using constraints
    {
        BlockDynamicSparsityPattern dsp(2, 2);
        dsp.block(0, 0).reinit(n_h, n_h);
        dsp.block(1, 0).reinit(n_q, n_h);
        dsp.block(0, 1).reinit(n_h, n_q);
        dsp.block(1, 1).reinit(n_q, n_q);

        dsp.collect_sizes();

        Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);

        for (unsigned int c = 0; c < dim + 1; ++c)
          for (unsigned int d = 0; d < dim + 1; ++d)
            if (!((c == dim) && (d == dim)))
              coupling[c][d] = DoFTools::always;
            else
              coupling[c][d] = DoFTools::none;

        DoFTools::make_sparsity_pattern(
          dof_handler, coupling, dsp, constraints, false);

        sparsity_pattern.copy_from(dsp);
    }
    
    {
        BlockDynamicSparsityPattern unconstrained_dsp(2, 2);
        unconstrained_dsp.block(0, 0).reinit(n_h, n_h);
        unconstrained_dsp.block(1, 0).reinit(n_q, n_h);
        unconstrained_dsp.block(0, 1).reinit(n_h, n_q);
        unconstrained_dsp.block(1, 1).reinit(n_q, n_q);

        unconstrained_dsp.collect_sizes();

        Table<2, DoFTools::Coupling> unconstrained_coupling(dim + 1, dim + 1);

        for (unsigned int c = 0; c < dim + 1; ++c)
          for (unsigned int d = 0; d < dim + 1; ++d)
            if (!((c == dim) && (d == dim)))
              unconstrained_coupling[c][d] = DoFTools::always;
            else
              unconstrained_coupling[c][d] = DoFTools::none;

        DoFTools::make_sparsity_pattern(dof_handler, 
                                        unconstrained_coupling, 
                                        unconstrained_dsp);

        unconstrained_sparsity_pattern.copy_from(unconstrained_dsp);
    }
    
    
    {
        BlockDynamicSparsityPattern preconditioner_dsp(2, 2);
        preconditioner_dsp.block(0, 0).reinit(n_h, n_h);
        preconditioner_dsp.block(1, 0).reinit(n_q, n_h);
        preconditioner_dsp.block(0, 1).reinit(n_h, n_q);
        preconditioner_dsp.block(1, 1).reinit(n_q, n_q);

        preconditioner_dsp.collect_sizes();

        Table<2, DoFTools::Coupling> preconditioner_coupling(dim + 1, dim + 1);

        for (unsigned int c = 0; c < dim + 1; ++c)
          for (unsigned int d = 0; d < dim + 1; ++d)
            if (((c == dim) && (d == dim)))
              preconditioner_coupling[c][d] = DoFTools::always;
            else
              preconditioner_coupling[c][d] = DoFTools::none;

        DoFTools::make_sparsity_pattern(dof_handler,
                                      preconditioner_coupling,
                                      preconditioner_dsp,
                                      constraints,
                                      false);

        preconditioner_sparsity_pattern.copy_from(preconditioner_dsp);
    }
   

    system_matrix.reinit(sparsity_pattern);
    preconditioner_matrix.reinit(preconditioner_sparsity_pattern);
    mass_p_matrix.reinit(preconditioner_sparsity_pattern.block(1,1));
    unconstrained_mass_matrix.reinit(unconstrained_sparsity_pattern);
    unconstrained_system_matrix.reinit(unconstrained_sparsity_pattern);

    previous_solution.reinit(2);
    previous_solution.block(0).reinit(n_h);
    previous_solution.block(1).reinit(n_q);
    previous_solution.collect_sizes();

    current_solution.reinit(2);
    current_solution.block(0).reinit(n_h);
    current_solution.block(1).reinit(n_q);
    current_solution.collect_sizes();
    
    system_rhs.reinit(2);
    system_rhs.block(0).reinit(n_h);
    system_rhs.block(1).reinit(n_q);
    system_rhs.collect_sizes();

    load_vector.reinit(2);
    load_vector.block(0).reinit(n_h);
    load_vector.block(1).reinit(n_q);
    load_vector.collect_sizes();

    // use exact solution to set up initial condition
    VectorTools::project(dof_handler,
                         constraints,
                         QGauss<dim>(fe.degree + 2),
                         exact_solution,
                         current_solution);
    
}

template <int dim>
void Maxwell<dim>::assemble_system()
{
    system_matrix         = 0;
    preconditioner_matrix = 0;
    mass_p_matrix         = 0;

    QGauss<dim> quadrature_formula(degree+2);

    MappingQGeneric<dim> mapping(1);
    FEValues<dim> fe_values(mapping,
                            fe,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                            update_gradients | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_system_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_preconditioner_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    Tensor<1, dim>              phi_i_h, phi_j_h;
    double                      phi_i_q, phi_j_q;
    double                      curl_phi_i_h, curl_phi_j_h;
    double                      vel_cross_phi_j_h;
    double                      div_phi_i_h, div_phi_j_h;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector h(0);
    const FEValuesExtractors::Scalar q(dim);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_mass_matrix = 0;
      cell_system_matrix = 0;
      cell_preconditioner_matrix = 0;
      fe_values.reinit(cell);

      for (unsigned int q_index = 0; q_index < n_q_points;
             ++q_index)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              phi_i_h = fe_values[h].value(i, q_index);
              phi_i_q = fe_values[q].value(i, q_index);
              curl_phi_i_h = fe_values[h].curl(i, q_index)[0];
              div_phi_i_h  = fe_values[h].divergence(i,q_index);

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  phi_j_h = fe_values[h].value(j, q_index);
                  phi_j_q = fe_values[q].value(j, q_index);
                  curl_phi_j_h = fe_values[h].curl(j,q_index)[0];
                  div_phi_j_h  = fe_values[h].divergence(j,q_index);

                  // mass matrix
                  cell_mass_matrix(i, j) +=
                        phi_i_h * phi_j_h * 
                        fe_values.JxW(q_index);

                  // pressure mass matrix aka preconditioner
                  cell_preconditioner_matrix(i, j) +=
                        phi_i_q * phi_j_q * 
                        fe_values.JxW(q_index);

                  // u cross h
                  vel_cross_phi_j_h = 
                      vel_field[0]*phi_j_h[1] - vel_field[1]*phi_j_h[0];

                  // system matrix
                  cell_system_matrix(i,j) +=
                      ( nu * curl_phi_i_h * curl_phi_j_h -
                        curl_phi_i_h * vel_cross_phi_j_h 
                        + 
                        beta * phi_i_h[0] * div_phi_j_h 
                        - 
                        div_phi_i_h * phi_j_q -
                        phi_i_q * div_phi_j_h 
                        ) * fe_values.JxW(q_index);

                }
            }
        }

      cell_system_matrix *= time_step;
      cell_system_matrix.add(1.0, cell_mass_matrix);

      cell->get_dof_indices(local_dof_indices);

      constraints.distribute_local_to_global(
        cell_system_matrix, local_dof_indices, system_matrix);
      constraints.distribute_local_to_global(
        cell_preconditioner_matrix, local_dof_indices, preconditioner_matrix);

      unconstrained_system_matrix.add(local_dof_indices, cell_system_matrix);
      unconstrained_mass_matrix.add(local_dof_indices, cell_mass_matrix);

    }
    mass_p_matrix.copy_from(preconditioner_matrix.block(1,1));
}

template <int dim>
void Maxwell<dim>::run()
{
    setup_grid();
    setup_dofs();

    std::cout << "   Number of active cells:       "
            << triangulation.n_active_cells() << std::endl;
    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << '\n'
            << "   Timestep size: "
            << time_step
            << std::endl;

    assemble_system();

    
    const auto &A = system_matrix.block(0,0);
    const auto &B = system_matrix.block(1,0);
    const auto &B_T = system_matrix.block(0,1);
    Vector<double> tmp(system_rhs.block(0).size());
    Vector<double> schur_rhs(system_rhs.block(1).size());

    SparseDirectUMFPACK A_inv;
    A_inv.factorize(A);
    
    PreconditionSSOR<SparseMatrix<double>> preconditioner_M;
    preconditioner_M.initialize(mass_p_matrix); 

    // outer preconditioning
    // set schur complement preconditioner as inverse pressure mass matrix
    InverseMatrix<SparseMatrix<double>, 
                  PreconditionSSOR<SparseMatrix<double>> > 
            preconditioner_S(mass_p_matrix, preconditioner_M);

    SchurComplement<SparseDirectUMFPACK> schur_comp(system_matrix, A_inv);

    InverseMatrix<SchurComplement<SparseDirectUMFPACK>,
                  InverseMatrix<SparseMatrix<double>,
                  PreconditionSSOR<SparseMatrix<double>> > >
              S_inv(schur_comp, preconditioner_S);

    MappingQGeneric<dim> mapping(1);
    ForcingFunction<dim> forcing_function(nu, vel_field);
    ExactSolution<dim> exact_solution;

    for (; current_time <= 1; current_time += time_step, ++time_step_number)
    {
      // The current solution should be swapped with the previous solution:
      std::swap(previous_solution, current_solution);

      // Set up M h^k + dt f^{k + 1}
      {
        forcing_function.set_time(current_time + time_step);
        VectorTools::create_right_hand_side(mapping,
                                            dof_handler,
                                            QGauss<dim>(degree + 2),
                                            forcing_function,
                                            load_vector);
        load_vector *= time_step;
        unconstrained_mass_matrix.vmult_add(load_vector, previous_solution);
      }

      // setup constraints for imposing boundary conditions
      {
        constraints.clear();
        exact_solution.set_time(current_time + time_step);
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        
        // FE_Nedelec boundary condition.
        VectorTools::project_boundary_values_curl_conforming_l2(
            dof_handler,
            0,
            exact_solution,
            0,
            constraints,
            StaticMappingQ1<dim>::mapping);
    
        FEValuesExtractors::Scalar q(dim);

        // Lagrange multilier boundary condition.
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 0,
                                                 exact_solution,
                                                 constraints,
                                                 fe.component_mask(q));
      }
      constraints.close();

      // Now we want to set up C^T (b - A k)

      auto u_system_operator = block_operator(unconstrained_system_matrix);
      auto setup_constrained_rhs = constrained_right_hand_side(
          constraints, u_system_operator, load_vector);

      setup_constrained_rhs.apply(system_rhs);  

      const auto &F = system_rhs.block(0);

      // compute schur_rhs
      A_inv.vmult(tmp, F);
      B.vmult(schur_rhs, tmp);
      schur_rhs -= system_rhs.block(1);

      // solve for q
      S_inv.vmult(current_solution.block(1), schur_rhs);
      constraints.distribute(current_solution);

      // compute second system rhs
      B_T.vmult(tmp, current_solution.block(1));
      tmp *= -1;
      tmp += F; 

      // solve for h
      A_inv.vmult(current_solution.block(0), tmp);
      constraints.distribute(current_solution);
  }

  output_results();
}

template <int dim>
void Maxwell<dim>::output_results() const
{
    std::vector<std::string> solution_names(dim, "magnetic_field");
    solution_names.emplace_back("lagrange_multiplier");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
           dim, DataComponentInterpretation::component_is_part_of_vector); 
    data_component_interpretation.push_back(
           DataComponentInterpretation::component_is_scalar);

    // Compute the pointwise maximum error:
    Vector<double> max_error_per_cell(triangulation.n_active_cells());
    {
        ExactSolution<dim> exact_solution;
        exact_solution.set_time(current_time);

        MappingQGeneric<dim> mapping(1);
        VectorTools::integrate_difference(mapping,
                                          dof_handler,
                                          current_solution,
                                          exact_solution,
                                          max_error_per_cell,
                                          QIterated<dim>(QGauss<1>(2), 2),
                                          VectorTools::NormType::Linfty_norm);
        std::cout << "\t\t\t\tmaximum error = "
                  << *std::max_element(max_error_per_cell.begin(),
                                       max_error_per_cell.end())
                  << std::endl;
        std::cout << "\t\t\t\t               ("
                  << *std::max_element(max_error_per_cell.begin(),
                                       max_error_per_cell.end())/4
                  << ")" 
                  << std::endl;
    }

    // Save the output:
    {
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(current_solution, 
                                 solution_names,
                                 DataOut<dim>::type_dof_data,
                                 data_component_interpretation);
        data_out.add_data_vector(max_error_per_cell, "max_error_per_cell");
        data_out.build_patches();

        std::ofstream output("maxwell_output-" + std::to_string(dim) + "d-" +
                             std::to_string(n_global_refinements) + ".vtu");
        data_out.write_vtu(output);
    }
}


int
main()
{
  int degree = 1;

  Tensor<1, 2> velocity_field;
  velocity_field[0] = 1.0;
  velocity_field[1] = 1.0;

  for (unsigned int i = 0; i < 6; ++i)
    {
      Maxwell<2> maxwell(degree, i, velocity_field);
      maxwell.run();
    }

  return 0;
}
