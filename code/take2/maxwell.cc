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

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

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
        void assemble_rhs();
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
        //BlockSparsityPattern      unconstrained_sparsity_pattern;
        BlockSparseMatrix<double> system_matrix;
        //BlockSparseMatrix<double> mass_matrix;

        //BlockSparseMatrix<double> unconstrained_system_matrix;
        //BlockSparseMatrix<double> unconstrained_mass_matrix;

        //BlockSparsityPattern      preconditioner_sparsity_pattern;
        //BlockSparseMatrix<double> preconditioner_matrix;

        BlockVector<double> current_solution;
        BlockVector<double> previous_solution;
        //BlockVector<double> system_rhs;
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
Maxwell<dim>::Maxwell(const unsigned int degree,
                      const unsigned int n_global_refinements,
                      const Tensor<1, dim> velocity_field)
    : degree(degree)
    , time_step(std::pow(0.1, n_global_refinements))
    , current_time(0.0)
    , time_step_number(1)
    , nu(1.0)
    , beta(1.0)
    , vel_field(velocity_field)
    , fe(FE_NedelecSZ<dim>(degree+1), dim, FE_Q<dim>(degree), 1)
    //, fe(FE_NedelecSZ<dim>(degree+1), dim, FE_Q<dim>(degree), 1)
    , dof_handler(triangulation)
{}

template <int dim>
void
Maxwell<dim>::setup_grid()
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
void Maxwell<dim>::setup_dofs()
{
    // idk if i need these?
    //A_preconditioner.reset();
    system_matrix.clear();
    //unconstrained_mass_matrix.clear();
    //unconstrained_system_matrix.clear();
    //preconditioner_matrix.clear();

    dof_handler.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(dof_handler);

    // group together the velocity components 
    // separate from the lagrange multiplier
    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    std::cout << "HERE I AM" << std::endl;
    std::cout << dof_handler.n_locally_owned_dofs() << std::endl;

    const hp::FECollection<dim, dim> fe_collection(
      dof_handler.get_fe_collection());

    std::cout << fe_collection.n_components() << std::endl;

    DoFRenumbering::component_wise(dof_handler, block_component);
    //DoFRenumbering::component_wise(dof_handler);
    std::cout << "I made it!" << std::endl;

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
    std::vector<types::global_dof_index> dofs_per_block;
    DoFTools::count_dofs_per_block(dof_handler, 
                                   dofs_per_block,
                                   block_component);
    const unsigned int n_h = dofs_per_block[0];
    const unsigned int n_q = dofs_per_block[1];

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
    /*
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
    */
    /*
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
    */

    system_matrix.reinit(sparsity_pattern);
    //preconditioner_matrix.reinit(preconditioner_sparsity_pattern);
    //unconstrained_mass_matrix.reinit(unconstrained_sparsity_pattern);
    //unconstrained_system_matrix.reinit(unconstrained_sparsity_pattern);

    previous_solution.reinit(2);
    previous_solution.block(0).reinit(n_h);
    previous_solution.block(1).reinit(n_q);
    previous_solution.collect_sizes();

    current_solution.reinit(2);
    current_solution.block(0).reinit(n_h);
    current_solution.block(1).reinit(n_q);
    current_solution.collect_sizes();
    
    /*
    system_rhs.reinit(2);
    system_rhs.block(0).reinit(n_h);
    system_rhs.block(1).reinit(n_q);
    system_rhs.collect_sizes();
    */

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
    //preconditioner_matrix = 0;

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
    //FullMatrix<double> cell_preconditioner_matrix(dofs_per_cell, dofs_per_cell);
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

                  // u cross h
                  vel_cross_phi_j_h = 
                      vel_field[0]*phi_j_h[1] - vel_field[1]*phi_j_h[0];

                  cell_system_matrix(i,j) +=
                      ( nu * curl_phi_i_h * curl_phi_j_h -
                        curl_phi_i_h * vel_cross_phi_j_h + 
                        beta * phi_i_h[0] * div_phi_j_h - 
                        div_phi_i_h * phi_j_q -
                        phi_i_q * div_phi_j_h ) * fe_values.JxW(q_index);
                }
            }
        }

      cell_system_matrix *= time_step;
      cell_system_matrix.add(1.0, cell_mass_matrix);

      // set up the three matrices we care about.
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        cell_system_matrix, local_dof_indices, system_matrix);
      //unconstrained_system_matrix.add(local_dof_indices, cell_system_matrix);
      //unconstrained_mass_matrix.add(local_dof_indices, cell_mass_matrix);
    }
}

template <int dim>
void Maxwell<dim>::assemble_rhs()
{

    load_vector = 0;

    QGauss<dim> quadrature_formula(degree+2);

    MappingQGeneric<dim> mapping(1);
    FEValues<dim> fe_values(mapping,
                            fe,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                            update_gradients | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    Vector<double>     cell_rhs(dofs_per_cell);

    ForcingFunction<dim>  forcing;
    forcing.set_time(current_time + time_step);
    std::vector<Vector<double>> forcing_values(n_q_points, Vector<double>(dim+1));

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector h(0);
    const FEValuesExtractors::Scalar q(dim);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_rhs = 0;
      fe_values.reinit(cell);

      forcing.vector_value_list(fe_values.get_quadrature_points(), 
                                forcing_values);

      for (unsigned int q_index = 0; q_index < n_q_points;
             ++q_index)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              //phi_i_h = fe_values[h].value(i, q_index);
              //cell_rhs(i) += phi_i_h * rhs_values[q] * fe_values.JxW(q);
              const unsigned int component_i = 
                  fe.system_to_component_index(i).first;
              cell_rhs(i) += fe_values.shape_value(i,q_index) *
                             forcing_values[q_index](component_i) *
                             fe_values.JxW(q_index);
            }
        }


      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
              cell_rhs, local_dof_indices, load_vector);
    }
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
    //ForcingFunction<dim> forcing_function;
    ExactSolution<dim> exact_solution;

    for (; current_time <= 1; current_time += time_step, ++time_step_number)
    {
      // The current solution should be swapped with the previous solution:
      std::swap(previous_solution, current_solution);

      // Set up M h^k + dt f^{k + 1}

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
      assemble_rhs();


      //auto u_system_operator = linear_operator(unconstrained_system_matrix);
      //auto setup_constrained_rhs = constrained_right_hand_side(
      //    constraints, u_system_operator, load_vector);

      // hopefully this deletes previous rhs entry maybe not though
      // stores the constrained rhs in rhs
      //setup_constrained_rhs.apply(system_rhs);  
      const auto &F = load_vector.block(0);

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
void Maxwell<dim>::output_results() const
{
    std::vector<std::string> solution_names(dim, "magnetic field");
    solution_names.emplace_back("lagrange multiplier");

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

  for (unsigned int i = 1; i < 5; ++i)
    {
      Maxwell<2> maxwell(degree, i, velocity_field);
      maxwell.run();
    }

  return 0;
}
