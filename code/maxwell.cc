#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nedelec_sz.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/constrained_linear_operator.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
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

// This program solves 
//
//     h_t + nu curl curl h - curl ( u x h ) = f
//
// with the backward Euler method, discretized with the method of lines:
//
//     M h^{k + 1} - M h^{k} = -dt S1 u^{k + 1} + dt S2 u^{k+1} + dt f^{k + 1}
//
// solving for u^{k + 1} yields
//
//     (M + dt S1 - dt S2) u^{k + 1} = M u^{k} + dt f^{k + 1}
//
// here M is the mass matrix and S1 is the discretization of the curl curl operator
// and S2 is the discretization of curl (u x ) operator

template <int dim>
class ForcingFunction : public Function<dim>
{
public:
  ForcingFunction()
    : Function<dim>(2)
  {}

  virtual void 
  vector_value(const Point<dim> &p, Vector<double> &  values) const override
  {
    const double time = this->get_time();
    //values(0) = -(p(0)+1)*sin(time);
    //values(1) = (p(1)-1)*cos(time);
    values(0) = (1+time*(1+p(0)*p(0)-p(0)))*cos(p(0)*p(1)) - (p(1)+1)*time*p(0)*sin(p(0)*p(1));
    values(1) = (1+time*(-1+p(1)*p(1)+p(1)))*sin(p(0)*p(1)) - (p(0)-1)*time*p(1)*cos(p(0)*p(1));
  }
};


template <int dim>
class ExactSolution : public Function<dim>
{
public:
  ExactSolution()
    : Function<dim>(2)
  {}

  virtual void 
  vector_value(const Point<dim> &p, Vector<double> &  values) const override
  {
    const double time = this->get_time();
    
    //values(0) = p(0)*cos(time);
    //values(1) = p(1)*sin(time);
    values(0) = time*cos(p(0)*p(1));
    values(1) = time*sin(p(0)*p(1));
  }
};


template <int dim>
class Maxwell
{
public:
  Maxwell(const unsigned int n_global_refinements,  const Tensor<1, dim> velocity_field);

  void
  run();

private:
  void
  setup_grid();
  void
  setup_system();
  void
  assemble_system();
  void
  do_one_time_step();
  void
  output_results() const;

  unsigned int n_global_refinements;
  double       time_step;
  int          time_step_number;
  double       nu;
  double       beta;

  Triangulation<dim> triangulation;
  double             current_time;

  FE_NedelecSZ<dim>       finite_element;
  DoFHandler<dim>       dof_handler;

  AffineConstraints<double> homogeneous_constraints;

  SparsityPattern      constrained_sparsity_pattern;
  SparsityPattern      unconstrained_sparsity_pattern;
  SparseMatrix<double> constrained_system_matrix;
  SparseMatrix<double> unconstrained_mass_matrix;
  SparseMatrix<double> unconstrained_system_matrix;

  Vector<double> previous_solution;
  Vector<double> current_solution;
  Tensor<1, dim> vel_field;
};



template <int dim>
Maxwell<dim>::Maxwell(const unsigned int n_global_refinements, const Tensor<1, dim> velocity_field)
  : n_global_refinements(n_global_refinements)
  , time_step(std::pow(0.1, n_global_refinements))
  , nu(1.0)
  , beta(1.0)
  , current_time(0.0)
  , finite_element(1)
  , dof_handler(triangulation)
  , vel_field(velocity_field)
{}



template <int dim>
void
Maxwell<dim>::setup_grid()
{
  GridGenerator::hyper_cube(triangulation);

  std::vector<GridTools::PeriodicFacePair<
      typename parallel::distributed::Triangulation<dim>::cell_iterator>>
      periodicity_vector;
  GridTools::collect_periodic_faces(triangulation,
                                      2,
                                      3,
                                      1,
                                      periodicity_vector,
                                      Tensor<1, dim>(),
                                      rotation_matrix);

  triangulation.refine_global(n_global_refinements);

  std::ofstream out("grid-1.svg");
  GridOut grid_out;
  grid_out.write_svg(triangulation,out);
  std::cout << "wrote the grid to grid-1.svg" << std::endl;
}



template <int dim>
void
Maxwell<dim>::setup_system()
{
  dof_handler.distribute_dofs(finite_element);

  previous_solution.reinit(dof_handler.n_dofs());
  current_solution.reinit(dof_handler.n_dofs());
  //
  ExactSolution<dim> exact_solution;
  exact_solution.set_time(current_time);
  // FE_Nedelec boundary condition.
  VectorTools::project_boundary_values_curl_conforming_l2(
    dof_handler,
    0,
    exact_solution,
    0,
    homogeneous_constraints,
    StaticMappingQ1<dim>::mapping);

  homogeneous_constraints.close();

  // use exact solution to set up initial condition
  VectorTools::project(dof_handler,
                       homogeneous_constraints,
                       QGauss<dim>(finite_element.degree + 2),
                       exact_solution,
                       current_solution);

  // For this problem we have to solve a more complicated situation with
  // constraints - in general, with the method of manufactured solutions, our
  // boundary conditions may be time dependent. However, we still want to know
  // what our constraints are when we set up the linear system. Hence we set up
  // the linear system with homogeneous constraints and then apply the correct
  // inhomogeneities when we solve the system.
  

  // set up the sparsity pattern, mass, and system matrices here.
  DynamicSparsityPattern constrained_dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, 
                                  constrained_dsp,
                                  homogeneous_constraints,
                                  /* keep_constrained_dofs */ false);
  DynamicSparsityPattern unconstrained_dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, unconstrained_dsp);

  constrained_sparsity_pattern.copy_from(constrained_dsp);
  unconstrained_sparsity_pattern.copy_from(unconstrained_dsp);

  constrained_system_matrix.reinit(constrained_sparsity_pattern);
  unconstrained_mass_matrix.reinit(unconstrained_sparsity_pattern);
  unconstrained_system_matrix.reinit(unconstrained_sparsity_pattern);
}

template <int dim>
void
Maxwell<dim>::assemble_system()
{
  QGauss<dim> quadrature(finite_element.degree + 2);

  MappingQGeneric<dim> mapping(1);
  FEValues<dim> fe_values(mapping,
                          finite_element,
                          quadrature,
                          update_values | update_gradients | 
                          update_quadrature_points | update_JxW_values);

  FEValuesViews::Vector<dim> fe_views(fe_values, 0);

  const unsigned int dofs_per_cell = finite_element.n_dofs_per_cell();

  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_system_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  double  cross_product;
  double curl_curl;
  double curl_cross;
  double dot_product;
  Tensor<1, dim>              value_i, value_j;

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_mass_matrix = 0;
      cell_system_matrix = 0;
      fe_values.reinit(cell);

      for (unsigned int q_index = 0; q_index < quadrature.size();
             ++q_index)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              value_i[0] = fe_values.shape_value_component(i,q_index,0);
              value_i[1] = fe_values.shape_value_component(i,q_index,1);
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  value_j[0] = fe_values.shape_value_component(j,q_index,0);
                  value_j[1] = fe_values.shape_value_component(j,q_index,1);

                  dot_product = value_i[0]*value_j[0]+value_i[1]*value_j[1];
                  
                  // mass matrix
                  cell_mass_matrix(i, j) +=
                        dot_product *
                        fe_values.JxW(q_index);

                  // u cross h
                  cross_product = vel_field[0]*fe_values.shape_value_component(j,q_index,1) 
                      - vel_field[1]*fe_values.shape_value_component(j,q_index,0);
                  beta_term = value_i[0] * fe_views.divergence(j,q_index)[0];
                  curl_curl = fe_views.curl(i,q_index)[0]* fe_views.curl(j,q_index)[0];
                  curl_cross = fe_views.curl(i, q_index)[0] * cross_product;
                  cell_system_matrix(i,j) +=
                        (nu * curl_curl - curl_cross + beta*beta_term) * 
                        fe_values.JxW(q_index);
                }
            }
        }

      cell_system_matrix *= time_step;
      cell_system_matrix.add(1.0, cell_mass_matrix);

      // set up the three matrices we care about.
      cell->get_dof_indices(local_dof_indices);
      homogeneous_constraints.distribute_local_to_global(
        cell_system_matrix, local_dof_indices, constrained_system_matrix);

      // To set up the RHS we need an unconstrained copy of the system matrix
      // and mass matrix too:
      /*
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
              unconstrained_system_matrix.add(local_dof_indices[i],
                                              local_dof_indices[j],
                                              cell_system_matrix(i,j));
                                              */
      unconstrained_system_matrix.add(local_dof_indices, cell_system_matrix);
      unconstrained_mass_matrix.add(local_dof_indices, cell_mass_matrix);
    }
}


template <int dim>
void
Maxwell<dim>::run()
{
  setup_grid();
  std::cout << "   Number of active cells:       "
            << triangulation.n_active_cells() << std::endl;

  setup_system();
  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << '\n'
            << "   Timestep size: "
            << time_step
            << std::endl;

  assemble_system();
  

  Vector<double> load_vector(dof_handler.n_dofs());
  MappingQGeneric<dim> mapping(1);
  ForcingFunction<dim> forcing_function;
  ExactSolution<dim> exact_solution;
  Vector<double> rhs(dof_handler.n_dofs());

  for (; current_time <= 1; current_time += time_step, ++time_step_number)
  {
      // The current solution should be swapped with the previous solution:
      std::swap(previous_solution, current_solution);

      // Set up M u^k + dt f^{k + 1}
      {
        forcing_function.set_time(current_time + time_step);
        VectorTools::create_right_hand_side(mapping,
                                            dof_handler,
                                            QGauss<dim>(finite_element.degree + 2),
                                            forcing_function,
                                            load_vector);  // prior content is deleted
        load_vector *= time_step;
        unconstrained_mass_matrix.vmult_add(load_vector, previous_solution);
      }

      // Similarly, at this point we can actually set up the correct constraints for
      // imposing the correct boundary values.
      //
      AffineConstraints<double> constraints;
      // configure constraints to interpolate the boundary values at the
      // correct time.
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

      // At this point we are done with finite element stuff and are working at a
      // purely algebraic level. We want to set up the RHS
      //
      //     C^T (b - A k)
      //
      // where b is the unconstrained system RHS, A is the unconstrained system
      // matrix, and k is a vector containing all the inhomogeneities of the linear
      // system. The PackedOperation object constrained_right_hand_side does all of
      // this for us.
      //
      // This code looks a little weird - the intended use case for
      // constrained_right_hand_side is to use the same load vector to set up
      // multiple system right hand sides, which we don't do here.
      //
      // Finally, deal.II supports chaining together multiple matrices and
      // matrix-like operators by wrapping them with a LinearOperator class. We will
      // use this to implement our own Stokes solver later in the course. Here we
      // wrap unconstrained_system_matrix as a LinearOperator and then pass that to
      // setup_constrained_rhs, which only needs the action of the linear operator
      // and not access to matrix entries.
      auto u_system_operator = linear_operator(unconstrained_system_matrix);
      auto setup_constrained_rhs = constrained_right_hand_side(
          constraints, u_system_operator, load_vector);

      // hopefully this deletes previous rhs entry maybe not though
      setup_constrained_rhs.apply(rhs);  // stores the constrained rhs in rhs

      // set up the solver and preconditioner here. Based on what was done
      // above you should use *rhs* as the final right-hand side and the
      // *constrained_system_matrix* as the final matrix. Be sure to distribute
      // constraints once you are done.
      SolverControl solver_control(1000, 1e-12*rhs.l2_norm());
      SolverCG<Vector<double>> cg(solver_control);

      PreconditionSSOR<SparseMatrix<double>> preconditioner;
      preconditioner.initialize(constrained_system_matrix, 1.0);
      //PreconditionJacobi<> preconditioner;
      //preconditioner.initialize(constrained_system_matrix);

      cg.solve(constrained_system_matrix, current_solution, rhs, preconditioner);

      constraints.distribute(current_solution);

  }

  output_results();
  
}



template <int dim>
void
Maxwell<dim>::output_results() const
{
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
              << "\t ("
              << *std::max_element(max_error_per_cell.begin(),
                                   max_error_per_cell.end()) * 0.1
              << ")"
              << std::endl;
  }

  // Save the output:
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(current_solution, "solution");
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

  Tensor<1, 2> velocity_field;
  velocity_field[0] = 1.0;
  velocity_field[1] = 1.0;

  for (unsigned int i = 1; i < 5; ++i)
    {
      Maxwell<2> maxwell(i, velocity_field);
      maxwell.run();
    }

  return 0;
}
