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

using namespace dealii;

template <int dim>
class NavierStokes
{
    public:
        NavierStokes(const unsigned int degree, 
                const unsigned int n_global_refinements,
                const Tensor<1, dim> magnetic_field);


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
        double mu;
        double rho;
        double gamma;

        Tensor<1, dim> mag_field;

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

        BlockVector<double> current_solution;
        BlockVector<double> previous_solution;
        BlockVector<double> system_rhs;
        BlockVector<double> load_vector;

        // maybe BlockSchurPreconditioner instead
        //std::shared_ptr< SparseDirectUMFPACK > A_preconditioner;
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
            values(0) = (1+time*(1+p(0)*p(0)-p(0)))*cos(p(0)*p(1)) - (p(1)+1)*time*p(0)*sin(p(0)*p(1));
            values(1) = (1+time*(-1+p(1)*p(1)+p(1)))*sin(p(0)*p(1)) - (p(0)-1)*time*p(1)*cos(p(0)*p(1));
            values(2) = 0;
        }
};

template <int dim>
class ForcingFunction : public Function<dim>
{
    public:
        ForcingFunction()
            : Function<dim>(dim+1) // 3
        {}

        virtual void 
        vector_value(const Point<dim> &p, 
                     Vector<double> &  values) const override
        {
            const double time = this->get_time();
            values(0) = (1+time*(1+p(0)*p(0)-p(0)))*cos(p(0)*p(1)) - (p(1)+1)*time*p(0)*sin(p(0)*p(1));
            values(1) = (1+time*(-1+p(1)*p(1)+p(1)))*sin(p(0)*p(1)) - (p(0)-1)*time*p(1)*cos(p(0)*p(1));
            values(2) = 0;
        }
};

template <int dim>
NavierStokes<dim>::NavierStokes(const unsigned int degree,
                                const unsigned int n_global_refinements,
                                const Tensor<1, dim> magnetic_field)
    : degree(degree)
    , n_global_refinements(n_global_refinements)
    , time_step(std::pow(0.1, n_global_refinements))
    , current_time(0.0)
    , time_step_number(1)
    , nu(1.0)
    , mu(1.0)
    , rho(1.0)
    , gamma(1.0)
    , mag_field(magnetic_field)
    , fe(FE_Q<dim>(degree+1), dim, FE_Q<dim>(degree), 1)
    , dof_handler(triangulation)
{}

template <int dim>
void
NavierStokes<dim>::setup_grid()
{
  GridGenerator::hyper_cube(triangulation);

  //std::vector<GridTools::PeriodicFacePair<
  //    typename parallel::distributed::Triangulation<dim>::cell_iterator>>
  //    periodicity_vector;
  //GridTools::collect_periodic_faces(triangulation,
  //                                   2,
  //                                    3,
  //                                    1,
  //                                    periodicity_vector,
  //                                    Tensor<1, dim>(),
  //                                    rotation_matrix);

  triangulation.refine_global(n_global_refinements);

  std::ofstream out("grid-1.svg");
  GridOut grid_out;
  grid_out.write_svg(triangulation,out);
  std::cout << "wrote the grid to grid-1.svg" << std::endl;
}


template <int dim>
void NavierStokes<dim>::setup_dofs()
{
    // idk if i need these?
    //A_preconditioner.reset();
    system_matrix.clear();
    unconstrained_mass_matrix.clear();
    unconstrained_system_matrix.clear();
    preconditioner_matrix.clear();

    dof_handler.distribute_dofs(fe);

    //DoFRenumbering::Cuthill_McKee(dof_handler);

    // group together the velocity components 
    // separate from the lagrange multiplier
    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;

    //DoFRenumbering::component_wise(dof_handler, block_component);
    DoFRenumbering::component_wise(dof_handler);

    std::vector<types::global_dof_index> dofs_per_component(
            dim+1, types::global_dof_index(0));

    DoFTools::count_dofs_per_component(dof_handler,
                                       dofs_per_component, 
                                       true); // this means there will be no dublicates

    const unsigned int n_u = dofs_per_component[0];
    const unsigned int n_p = dofs_per_component[dim];

    ExactSolution<dim> exact_solution;
    exact_solution.set_time(current_time);

    // FE_Nedelec boundary condition.
    VectorTools::project_boundary_values_curl_conforming_l2(
        dof_handler,
        0,
        exact_solution,
        0,
        constraints,
        StaticMappingQ1<dim>::mapping);

    constraints.close();

    //const std::vector<types::global_dof_index> dofs_per_block =
    //  DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    
    //std::vector<types::global_dof_index> dofs_per_block;
    //DoFTools::count_dofs_per_block(dof_handler, 
    //                               dofs_per_block,
    //                               block_component);

    {
        BlockDynamicSparsityPattern dsp(2, 2);
        dsp.block(0, 0).reinit(n_u, n_u);
        dsp.block(1, 0).reinit(n_p, n_u);
        dsp.block(0, 1).reinit(n_u, n_p);
        dsp.block(1, 1).reinit(n_p, n_p);

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
        unconstrained_dsp.block(0, 0).reinit(n_u, n_u);
        unconstrained_dsp.block(1, 0).reinit(n_p, n_u);
        unconstrained_dsp.block(0, 1).reinit(n_u, n_p);
        unconstrained_dsp.block(1, 1).reinit(n_p, n_p);

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
        preconditioner_dsp.block(0, 0).reinit(n_u, n_u);
        preconditioner_dsp.block(1, 0).reinit(n_p, n_u);
        preconditioner_dsp.block(0, 1).reinit(n_u, n_p);
        preconditioner_dsp.block(1, 1).reinit(n_p, n_p);

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
    unconstrained_mass_matrix.reinit(unconstrained_sparsity_pattern);
    unconstrained_system_matrix.reinit(unconstrained_sparsity_pattern);

    previous_solution.reinit(2);
    previous_solution.block(0).reinit(n_u);
    previous_solution.block(1).reinit(n_p);
    previous_solution.collect_sizes();

    current_solution.reinit(2);
    current_solution.block(0).reinit(n_u);
    current_solution.block(1).reinit(n_p);
    current_solution.collect_sizes();
    
    
    system_rhs.reinit(2);
    system_rhs.block(0).reinit(n_u);
    system_rhs.block(1).reinit(n_p);
    system_rhs.collect_sizes();
    

    load_vector.reinit(2);
    load_vector.block(0).reinit(n_u);
    load_vector.block(1).reinit(n_p);
    load_vector.collect_sizes();

    // use exact solution to set up initial condition
    VectorTools::project(dof_handler,
                         constraints,
                         QGauss<dim>(fe.degree + 2),
                         exact_solution,
                         current_solution);
    
}

template <int dim>
void NavierStokes<dim>::assemble_system()
{
    system_matrix         = 0;
    preconditioner_matrix = 0;

    QGauss<dim> quadrature_formula(degree+2);

    MappingQGeneric<dim> mapping(1);
    FEValues<dim> fe_values(mapping,
                            fe,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                            update_gradients | update_JxW_values);

    //FEValuesViews::Vector<dim> fe_views(fe_values, 0);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_system_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_preconditioner_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    Tensor<1, dim>              phi_i_u, phi_j_u;
    Tensor<2, dim>              grad_phi_i_u, grad_phi_j_u;
    double                      phi_i_p, phi_j_p;
    double                      div_phi_i_u, div_phi_j_u;


    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector u(0);
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
              phi_i_u = fe_values[u].value(i, q_index);
              phi_i_p = fe_values[p].value(i, q_index);
              grad_phi_i_u = fe_values[u].gradient(i, q_index);
              div_phi_i_u  = fe_values[u].divergence(i,q_index);


              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  phi_j_u = fe_values[u].value(j, q_index);
                  phi_j_p = fe_values[p].value(j, q_index);
                  grad_phi_j_u = fe_values[u].gradient(j, q_index);
                  div_phi_j_u  = fe_values[u].divergence(j,q_index);

                  // mass matrix
                  cell_mass_matrix(i, j) +=
                        phi_i_u * phi_j_u * 
                        fe_values.JxW(q_index);
                  
                  // pressure mass matrix aka preconditioner
                  cell_preconditioner_matrix(i, j) +=
                        phi_i_p * phi_j_p *
                        fe_values.JxW(q_index);

                  cell_system_matrix(i,j) +=
                      ( mu * 
                        scalar_product(grad_phi_i_u, grad_phi_j_u) +
                       (nu + mu) * div_phi_i_u * div_phi_i_v -
                        div_phi_i_u * phi_j_p -
                        phi_i_p * div_phi_j_u ) * fe_values.JxW(q_index);
                }
            }
        }

      cell_system_matrix *= time_step;
      cell_system_matrix.add(1.0, cell_mass_matrix);

      // set up the three matrices we care about.
      cell->get_dof_indices(local_dof_indices);

      constraints.distribute_local_to_global(
        cell_system_matrix, local_dof_indices, system_matrix);
      constraints.distribute_local_to_global(
        cell_preconditioner_matrix, local_dof_indices, preconditioner_matrix);

      unconstrained_system_matrix.add(local_dof_indices, cell_system_matrix);
      unconstrained_mass_matrix.add(local_dof_indices, cell_mass_matrix);
    }
}

template <int dim>
void NavierStokes<dim>::run()
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

    const auto op_A = linear_operator(A);
    const auto op_B = linear_operator(B);

    // solve A inverse to define schur complement
    ReductionControl reduction_control_A(2000, 1.0e-18, 1.0e-10);
    SolverGMRES<Vector<double>> solver_A(reduction_control_A);

    PreconditionJacobi<SparseMatrix<double>> preconditioner_A;
    preconditioner_A.initialize(A);

    const auto op_A_inv = inverse_operator(op_A, solver_A, preconditioner_A);

    const auto op_S = op_B * op_A_inv * transpose_operator(op_B);
    const auto op_aS =
        op_B * linear_operator(preconditioner_A) * transpose_operator(op_B);

    // now precondition S to solve system
    IterationNumberControl iteration_number_control_aS(30, 1.0e-18);
    SolverGMRES<Vector<double>> solver_aS(iteration_number_control_aS);
    const auto preconditioner_S = 
        inverse_operator(op_aS, solver_aS, PreconditionIdentity());


    SolverControl solver_control_S(2000, 1.0e-12);// 1e-6*schur_rhs.l2_norm()
    SolverGMRES<Vector<double>> solver_S(solver_control_S);

    const auto op_S_inv = inverse_operator(op_S, solver_S, preconditioner_S);


    MappingQGeneric<dim> mapping(1);
    ForcingFunction<dim> forcing_function;
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
      constraints.clear();
      exact_solution.set_time(current_time + time_step);
      // FE_Nedelec boundary condition.
      VectorTools::project_boundary_values_curl_conforming_l2(
        dof_handler,
        0,
        exact_solution,
        0,
        constraints,
        StaticMappingQ1<dim>::mapping);
      constraints.close();

      // Now we want to set up C^T (b - A k)

      auto u_system_operator = block_operator(unconstrained_system_matrix);
      auto setup_constrained_rhs = constrained_right_hand_side(
          constraints, u_system_operator, load_vector);

      setup_constrained_rhs.apply(system_rhs);  

      const auto &F = system_rhs.block(0);

      const auto schur_rhs = op_B * op_A_inv * F;

      solver_S.solve(op_S, current_solution.block(1), schur_rhs, preconditioner_S);

      constraints.distribute(current_solution);

      op_A_inv.vmult(
              current_solution.block(0), 
              F - transpose_operator(op_B)*current_solution.block(1));

      constraints.distribute(current_solution);
  }

    //q = op_S_inv * schur_rhs;
    //h = op_M_inv * (F - transpose_operator(op_B) * q);

  output_results();

}

template <int dim>
void NavierStokes<dim>::output_results() const
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
        std::cout << "maximum error = "
                  << *std::max_element(max_error_per_cell.begin(),
                                       max_error_per_cell.end())
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

        std::ofstream output("output-" + std::to_string(dim) + "d-" +
                             std::to_string(n_global_refinements) + ".vtu");
        data_out.write_vtu(output);
    }
}


int
main()
{
  int degree = 2;

  Tensor<1, 2> velocity_field;
  velocity_field[0] = 1.0;
  velocity_field[1] = 1.0;

  for (unsigned int i = 1; i < 3; ++i)
    {
      NavierStokes<2> ns(degree, i, velocity_field);
      ns.run();
    }

  return 0;
}
